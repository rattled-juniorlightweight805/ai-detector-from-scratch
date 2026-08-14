#!/usr/bin/env python3
"""Collect licensed human-written text published no later than 2022.

The script appends numbered UTF-8 text files to ``data/human`` and records
document-level provenance and license information in ``data/meta.json``.

Example
-------
python scripts/collect_licensed_human_text.py \
    --data-dir data \
    --sources all \
    --target-per-source 3500 \
    --cutoff-date 2022-12-31

The collector is resumable. A source that already has the requested number of
samples is skipped, existing text hashes are not written again, and metadata is
saved after every source document.
"""

import argparse
import calendar
import datetime as dt
import gzip
import hashlib
import html
import http.client
import io
import json
import math
import os
import re
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from pathlib import Path


DEFAULT_CUTOFF = dt.date(2022, 12, 31)
DEFAULT_TARGET_WORDS = (50, 100, 250, 500, 1000)
DEFAULT_USER_AGENT = (
    "local-ai-detector-dataset/0.1 "
    "(noncommercial dataset research; https://sebastianraschka.com)"
)

CC0_URL = "https://creativecommons.org/publicdomain/zero/1.0/"
CC_BY_URL = "https://creativecommons.org/licenses/by/4.0/"
CC_BY_SA_URL = "https://creativecommons.org/licenses/by-sa/4.0/"

SOURCE_CATALOG = {
    "pmc": {
        "name": "PubMed Central Open Access Subset",
        "homepage": "https://pmc.ncbi.nlm.nih.gov/tools/openftlist/",
        "license_policy": "Only records explicitly marked CC0 or CC BY",
    },
    "plos": {
        "name": "PLOS",
        "homepage": "https://plos.org/",
        "license_policy": "Only PLOS records explicitly marked CC0 or CC BY",
    },
    "wikimedia": {
        "name": "English Wikipedia",
        "homepage": "https://en.wikipedia.org/",
        "license_policy": "CC BY-SA 4.0; exact pre-cutoff revision retained",
    },
    "stackexchange": {
        "name": "Stack Overflow via the Stack Exchange API",
        "homepage": "https://stackoverflow.com/",
        "license_policy": "Versioned CC BY-SA license based on contribution date",
    },
    "gutenberg": {
        "name": "Project Gutenberg",
        "homepage": "https://www.gutenberg.org/",
        "license_policy": "Only records marked public domain; boilerplate removed",
    },
    "openstax": {
        "name": "OpenStax",
        "homepage": "https://openstax.org/",
        "license_policy": "Only CC BY 4.0 repository snapshots at the cutoff",
    },
    "arxiv": {
        "name": "arXiv",
        "homepage": "https://arxiv.org/",
        "license_policy": "Only CC0, CC BY 4.0, or CC BY-SA 4.0 records",
    },
}

ALL_SOURCES = tuple(SOURCE_CATALOG)


class SourceDocument(
    namedtuple(
        "SourceDocumentFields",
        (
            "collection",
            "document_id",
            "title",
            "text",
            "source_url",
            "source_date",
            "authors",
            "license_name",
            "license_url",
            "extra",
        ),
    )
):
    __slots__ = ()

    def __new__(
        cls,
        collection,
        document_id,
        title,
        text,
        source_url,
        source_date,
        authors,
        license_name,
        license_url,
        extra=None,
    ):
        return super().__new__(
            cls,
            collection,
            document_id,
            title,
            text,
            source_url,
            source_date,
            authors,
            license_name,
            license_url,
            {} if extra is None else extra,
        )


def local_name(tag):
    return tag.rsplit("}", 1)[-1]


def clean_inline_text(value):
    value = html.unescape(value)
    value = value.replace("\u00a0", " ")
    value = re.sub(r"[ \t\r\f\v]+", " ", value)
    value = re.sub(r" *\n *", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def word_count(text):
    return len(re.findall(r"\S+", text))


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_date(value):
    if not value:
        return None
    match = re.match(r"(\d{4})-(\d{2})-(\d{2})", value)
    if not match:
        return None
    try:
        return dt.date(*(int(part) for part in match.groups()))
    except ValueError:
        return None


def evenly_spaced(items, limit):
    if len(items) <= limit:
        return items
    if limit == 1:
        return [items[len(items) // 2]]
    indices = [round(i * (len(items) - 1) / (limit - 1)) for i in range(limit)]
    return [items[index] for index in indices]


def chunk_document(
    text,
    document_id,
    max_chunks,
    target_words = DEFAULT_TARGET_WORDS,
):
    """Return nonoverlapping, varied-length chunks spread over a document."""
    words = clean_inline_text(text).split()
    if len(words) < min(target_words):
        return []

    seed = int(hashlib.sha256(document_id.encode("utf-8")).hexdigest()[:8], 16)
    start_index = seed % len(target_words)
    candidates = []
    offset = 0
    chunk_index = 1

    while len(words) - offset >= min(target_words):
        target = target_words[(start_index + chunk_index - 1) % len(target_words)]
        end = min(offset + target, len(words))
        if 0 < len(words) - end < min(target_words):
            end = len(words)
        chunk = " ".join(words[offset:end]).strip()
        if word_count(chunk) >= min(target_words):
            candidates.append((chunk_index, target, chunk))
        offset = end
        chunk_index += 1

    return evenly_spaced(candidates, max_chunks)


class ParagraphHTMLParser(HTMLParser):
    """Extract prose paragraphs while dropping code, navigation, and tables."""

    SKIP_TAGS = {
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "table",
        "figure",
        "math",
        "code",
        "pre",
        "svg",
        "sup",
        "noscript",
        "blockquote",
    }

    def __init__(self, paragraph_tags = None):
        super().__init__(convert_charrefs=True)
        self.paragraph_tags = paragraph_tags or {"p"}
        self.skip_depth = 0
        self.capture_depth = 0
        self.current = []
        self.paragraphs = []

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag in self.paragraph_tags:
            if self.capture_depth == 0:
                self.current = []
            self.capture_depth += 1
        elif tag == "br" and self.capture_depth:
            self.current.append("\n")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            if self.skip_depth:
                self.skip_depth -= 1
            return
        if self.skip_depth:
            return
        if tag in self.paragraph_tags and self.capture_depth:
            self.capture_depth -= 1
            if self.capture_depth == 0:
                paragraph = clean_inline_text("".join(self.current))
                if word_count(paragraph) >= 5:
                    self.paragraphs.append(paragraph)
                self.current = []

    def handle_data(self, data):
        if self.capture_depth and not self.skip_depth:
            self.current.append(data)


class HttpClient:
    def __init__(self, user_agent, request_delay):
        self.user_agent = user_agent
        self.request_delay = request_delay
        self.last_request_at = 0.0

    def _pace(self, extra_delay = 0.0):
        delay = max(self.request_delay, extra_delay)
        remaining = delay - (time.monotonic() - self.last_request_at)
        if remaining > 0:
            time.sleep(remaining)

    def bytes(
        self,
        url,
        *,
        headers = None,
        extra_delay = 0.0,
        attempts = 5,
    ):
        request_headers = {
            "User-Agent": self.user_agent,
            "Accept-Encoding": "identity",
        }
        if headers:
            request_headers.update(headers)

        for attempt in range(attempts):
            self._pace(extra_delay)
            request = urllib.request.Request(url, headers=request_headers)
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    payload = response.read()
                self.last_request_at = time.monotonic()
                return payload
            except urllib.error.HTTPError as error:
                self.last_request_at = time.monotonic()
                if error.code not in {429, 500, 502, 503, 504} or attempt == attempts - 1:
                    raise
                retry_after = error.headers.get("Retry-After")
                wait = float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
                time.sleep(min(wait, 60))
            except (
                urllib.error.URLError,
                http.client.IncompleteRead,
                ConnectionError,
                TimeoutError,
            ):
                self.last_request_at = time.monotonic()
                if attempt == attempts - 1:
                    raise
                time.sleep(min(2**attempt, 30))
        raise RuntimeError(f"Unable to retrieve {url}")

    def json(
        self,
        url,
        *,
        headers = None,
        extra_delay = 0.0,
    ):
        return json.loads(self.bytes(url, headers=headers, extra_delay=extra_delay))


class DatasetWriter:
    def __init__(self, data_dir, cutoff, max_chunks):
        self.data_dir = data_dir
        self.human_dir = data_dir / "human"
        self.meta_path = data_dir / "meta.json"
        self.cutoff = cutoff
        self.max_chunks = max_chunks
        self.human_dir.mkdir(parents=True, exist_ok=True)

        if self.meta_path.exists():
            self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        else:
            self.meta = {
                "schema_version": 1,
                "counts": {"total": 0, "human": 0, "ai": 0},
                "samples": [],
            }

        self.meta.setdefault("source_catalog", {})
        self.meta["source_catalog"].update(SOURCE_CATALOG)
        self.meta["licensed_source_cutoff"] = cutoff.isoformat()
        self.samples = self.meta.setdefault("samples", [])
        self.hashes = {sample.get("sha256") for sample in self.samples if sample.get("sha256")}
        self.next_id = max((int(sample["id"]) for sample in self.samples), default=0) + 1
        self.counts = Counter(
            sample.get("collection") for sample in self.samples if sample.get("label") == "human"
        )

    def source_count(self, collection):
        return self.counts[collection]

    def append_document(self, document, target_total):
        if self.source_count(document.collection) >= target_total:
            return 0
        source_date = parse_date(document.source_date)
        if source_date and source_date > self.cutoff:
            raise ValueError(
                f"{document.document_id} is dated {source_date}, after {self.cutoff}"
            )

        chunks = chunk_document(document.text, document.document_id, self.max_chunks)
        added = 0
        for chunk_index, target_words, chunk in chunks:
            if self.source_count(document.collection) >= target_total:
                break
            digest = sha256_text(chunk)
            if digest in self.hashes:
                continue

            sample_id = self.next_id
            relative_file = f"human/{sample_id}.txt"
            output_path = self.data_dir / relative_file
            output_path.write_text(chunk.rstrip() + "\n", encoding="utf-8")

            sample = {
                "id": sample_id,
                "file": relative_file,
                "label": "human",
                "collection": document.collection,
                "source_name": SOURCE_CATALOG[document.collection]["name"],
                "source": document.document_id,
                "source_document_id": document.document_id,
                "title": document.title,
                "source_url": document.source_url,
                "source_date": document.source_date,
                "authors": document.authors,
                "license": document.license_name,
                "license_url": document.license_url,
                "sample_type": "chunk",
                "chunk_index": chunk_index,
                "target_words": target_words,
                "word_count": word_count(chunk),
                "sha256": digest,
                "retrieved_at": dt.datetime.now(dt.UTC).isoformat(),
            }
            sample.update(document.extra)
            self.samples.append(sample)
            self.hashes.add(digest)
            self.counts[document.collection] += 1
            self.next_id += 1
            added += 1

        if added:
            self.save()
        return added

    def save(self):
        human_count = sum(1 for sample in self.samples if sample.get("label") == "human")
        ai_count = sum(1 for sample in self.samples if sample.get("label") == "ai")
        self.meta["counts"] = {
            "total": human_count + ai_count,
            "human": human_count,
            "ai": ai_count,
        }
        temporary = self.meta_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.meta_path)


def build_url(base, parameters):
    return f"{base}?{urllib.parse.urlencode(parameters)}"


def extract_jats_text(xml_payload):
    root = ET.fromstring(xml_payload)
    unwanted = {
        "ref-list",
        "table-wrap",
        "fig",
        "disp-formula",
        "inline-formula",
        "supplementary-material",
        "media",
        "alternatives",
        "fn-group",
        "permissions",
    }
    blocked_sections = {
        "references",
        "bibliography",
        "supplementary material",
        "supporting information",
        "acknowledgments",
        "acknowledgements",
        "author contributions",
        "competing interests",
        "data availability",
    }

    paragraphs = []

    def direct_title(element):
        for child in element:
            if local_name(child.tag) == "title":
                return clean_inline_text(" ".join(child.itertext())).lower()
        return ""

    def walk(element):
        name = local_name(element.tag)
        if name in unwanted:
            return
        if name == "sec" and direct_title(element) in blocked_sections:
            return
        if name == "p":
            value = clean_inline_text(" ".join(element.itertext()))
            if word_count(value) >= 5:
                paragraphs.append(value)
            return
        for child in element:
            walk(child)

    for element in root.iter():
        if local_name(element.tag) == "abstract":
            walk(element)
    for element in root.iter():
        if local_name(element.tag) == "body":
            walk(element)
            break
    return "\n\n".join(paragraphs)


def epmc_license(record):
    value = re.sub(r"[^a-z0-9]+", " ", record.get("license", "").lower()).strip()
    if value in {"cc0", "cc zero"}:
        return "CC0 1.0", CC0_URL
    if value == "cc by":
        return "CC BY 4.0", CC_BY_URL
    return None


def is_plos_record(record):
    doi = record.get("doi", "").lower()
    journal = (
        record.get("journalInfo", {}).get("journal", {}).get("title", "").lower()
    )
    return doi.startswith("10.1371/") or journal.startswith("plos")


def collect_epmc_source(
    client,
    writer,
    target,
    cutoff,
    *,
    plos_only,
):
    collection = "plos" if plos_only else "pmc"
    if writer.source_count(collection) >= target:
        print(f"[{collection}] already has {writer.source_count(collection)} samples")
        return

    years = list(range(cutoff.year, 2002, -1))
    for year_index, year in enumerate(years):
        if writer.source_count(collection) >= target:
            break
        remaining = target - writer.source_count(collection)
        remaining_years = len(years) - year_index
        year_quota = math.ceil(remaining / remaining_years)
        added_this_year = 0

        query_parts = [
            "OPEN_ACCESS:Y",
            '(LICENSE:"cc by" OR LICENSE:"cc0")',
            f"FIRST_PDATE:[{year}-01-01 TO {year}-12-31]",
        ]
        if plos_only:
            query_parts.append('PUBLISHER:"Public Library of Science"')
        query = " AND ".join(query_parts)
        cursor = "*"

        while added_this_year < year_quota and writer.source_count(collection) < target:
            url = build_url(
                "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                {
                    "query": query,
                    "format": "json",
                    "resultType": "core",
                    "pageSize": 100,
                    "cursorMark": cursor,
                },
            )
            payload = client.json(url)
            records = payload.get("resultList", {}).get("result", [])
            if not records:
                break

            for record in records:
                if writer.source_count(collection) >= target or added_this_year >= year_quota:
                    break
                if not plos_only and is_plos_record(record):
                    continue
                license_info = epmc_license(record)
                pmcid = record.get("pmcid")
                publication_date = record.get("firstPublicationDate")
                if not license_info or not pmcid:
                    continue
                if parse_date(publication_date) and parse_date(publication_date) > cutoff:
                    continue

                xml_url = (
                    "https://www.ebi.ac.uk/europepmc/webservices/rest/"
                    f"{urllib.parse.quote(pmcid)}/fullTextXML"
                )
                try:
                    text = extract_jats_text(client.bytes(xml_url))
                except (ET.ParseError, urllib.error.URLError, urllib.error.HTTPError) as error:
                    print(f"[{collection}] skipped {pmcid}: {error}", file=sys.stderr)
                    continue

                authors = [
                    author.get("fullName", "").strip()
                    for author in record.get("authorList", {}).get("author", [])
                    if author.get("fullName")
                ]
                doi = record.get("doi")
                source_url = (
                    f"https://doi.org/{doi}"
                    if doi
                    else f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
                )
                document = SourceDocument(
                    collection=collection,
                    document_id=f"{collection}-{pmcid.lower()}",
                    title=clean_inline_text(record.get("title", pmcid)),
                    text=text,
                    source_url=source_url,
                    source_date=publication_date,
                    authors=authors,
                    license_name=license_info[0],
                    license_url=license_info[1],
                    extra={
                        "pmcid": pmcid,
                        "doi": doi,
                        "journal": record.get("journalInfo", {})
                        .get("journal", {})
                        .get("title"),
                        "access_api": "Europe PMC REST API",
                    },
                )
                added = writer.append_document(document, target)
                added_this_year += added
                if added:
                    print(
                        f"[{collection}] {writer.source_count(collection)}/{target} "
                        f"from {pmcid}"
                    )

            next_cursor = payload.get("nextCursorMark")
            if not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor


def wikipedia_revision_before(
    client, page_id, cutoff
):
    url = build_url(
        "https://en.wikipedia.org/w/api.php",
        {
            "action": "query",
            "prop": "revisions",
            "pageids": page_id,
            "rvprop": "ids|timestamp|size",
            "rvstart": f"{cutoff.isoformat()}T23:59:59Z",
            "rvdir": "older",
            "rvlimit": 1,
            "format": "json",
            "formatversion": 2,
        },
    )
    payload = client.json(url)
    pages = payload.get("query", {}).get("pages", [])
    if not pages or not pages[0].get("revisions"):
        return None
    revision = pages[0]["revisions"][0]
    revision["title"] = pages[0].get("title", str(page_id))
    return revision


def fetch_wikimedia_document(
    page,
    cutoff,
    min_revision_bytes,
    user_agent,
    request_delay,
):
    page_id = int(page["id"])
    title = page.get("title", "")
    client = HttpClient(user_agent, request_delay)
    try:
        revision = wikipedia_revision_before(client, page_id, cutoff)
        if not revision:
            return None
        revision_size = int(revision.get("size", 0))
        if revision_size < min_revision_bytes:
            return None
        revision_id = revision.get("revid")
        if not revision_id:
            return None
        revision_url = (
            "https://en.wikipedia.org/w/rest.php/v1/revision/"
            f"{revision_id}/with_html"
        )
        revision_payload = client.json(revision_url)
    except (urllib.error.URLError, urllib.error.HTTPError) as error:
        print(f"[wikimedia] skipped page {page_id}: {error}", file=sys.stderr)
        return None

    parser = ParagraphHTMLParser()
    parser.feed(revision_payload.get("html", ""))
    text = "\n\n".join(parser.paragraphs)
    return SourceDocument(
        collection="wikimedia",
        document_id=f"wikipedia-{page_id}-rev-{revision_id}",
        title=revision.get("title", title),
        text=text,
        source_url=(
            "https://en.wikipedia.org/w/index.php?"
            + urllib.parse.urlencode({"curid": page_id, "oldid": revision_id})
        ),
        source_date=revision.get("timestamp", "")[:10] or None,
        authors=["Wikipedia contributors"],
        license_name="CC BY-SA 4.0",
        license_url=CC_BY_SA_URL,
        extra={
            "page_id": page_id,
            "revision_id": revision_id,
            "revision_size_bytes": revision_size,
            "attribution_url": (
                "https://en.wikipedia.org/w/index.php?"
                + urllib.parse.urlencode({"title": title, "action": "history"})
            ),
        },
    )


def collect_wikimedia(
    client,
    writer,
    target,
    cutoff,
    min_revision_bytes,
    workers,
):
    collection = "wikimedia"
    if writer.source_count(collection) >= target:
        print(f"[{collection}] already has {writer.source_count(collection)} samples")
        return

    seen_page_ids = set()
    attempts = 0
    max_attempts = target * 3
    with ThreadPoolExecutor(max_workers=workers) as executor:
        while writer.source_count(collection) < target and attempts < max_attempts:
            random_url = build_url(
                "https://en.wikipedia.org/w/api.php",
                {
                    "action": "query",
                    "list": "random",
                    "rnnamespace": 0,
                    "rnlimit": 50,
                    "format": "json",
                    "formatversion": 2,
                },
            )
            pages = client.json(random_url).get("query", {}).get("random", [])
            if not pages:
                break

            candidates = []
            for page in pages:
                page_id = int(page["id"])
                title = page.get("title", "")
                attempts += 1
                if page_id in seen_page_ids or title.lower().startswith("list of "):
                    continue
                seen_page_ids.add(page_id)
                candidates.append(page)

            futures = [
                executor.submit(
                    fetch_wikimedia_document,
                    page,
                    cutoff,
                    min_revision_bytes,
                    client.user_agent,
                    client.request_delay,
                )
                for page in candidates
            ]
            for future in as_completed(futures):
                if writer.source_count(collection) >= target:
                    break
                document = future.result()
                if not document:
                    continue
                added = writer.append_document(document, target)
                if added:
                    print(
                        f"[wikimedia] {writer.source_count(collection)}/{target} "
                        f"from {document.title}"
                    )

    if writer.source_count(collection) < target:
        raise RuntimeError(
            f"Wikimedia stopped after {attempts} page attempts with "
            f"{writer.source_count(collection)}/{target} samples"
        )


def stackexchange_license(created_at):
    if created_at < dt.date(2011, 4, 8):
        return (
            "CC BY-SA 2.5",
            "https://creativecommons.org/licenses/by-sa/2.5/",
        )
    if created_at < dt.date(2018, 5, 2):
        return (
            "CC BY-SA 3.0",
            "https://creativecommons.org/licenses/by-sa/3.0/",
        )
    return "CC BY-SA 4.0", CC_BY_SA_URL


def collect_stackexchange(
    client, writer, target, cutoff
):
    collection = "stackexchange"
    if writer.source_count(collection) >= target:
        print(f"[{collection}] already has {writer.source_count(collection)} samples")
        return

    cutoff_timestamp = calendar.timegm(
        dt.datetime.combine(cutoff, dt.time(23, 59, 59), tzinfo=dt.UTC).utctimetuple()
    )
    for year in range(cutoff.year, 2007, -1):
        if writer.source_count(collection) >= target:
            break
        year_start = dt.date(year, 1, 1)
        year_end = min(dt.date(year, 12, 31), cutoff)
        from_timestamp = calendar.timegm(
            dt.datetime.combine(year_start, dt.time.min, tzinfo=dt.UTC).utctimetuple()
        )
        to_timestamp = calendar.timegm(
            dt.datetime.combine(year_end, dt.time(23, 59, 59), tzinfo=dt.UTC).utctimetuple()
        )

        for page in range(1, 26):
            if writer.source_count(collection) >= target:
                break
            url = build_url(
                "https://api.stackexchange.com/2.3/answers",
                {
                    "site": "stackoverflow",
                    "page": page,
                    "pagesize": 100,
                    "order": "desc",
                    "sort": "votes",
                    "fromdate": from_timestamp,
                    "todate": to_timestamp,
                    "filter": "withbody",
                },
            )
            payload = client.json(url)
            backoff = payload.get("backoff")
            if backoff:
                time.sleep(min(int(backoff), 60))

            for answer in payload.get("items", []):
                if writer.source_count(collection) >= target:
                    break
                created_timestamp = answer.get("creation_date")
                last_activity = answer.get("last_activity_date", created_timestamp)
                if not created_timestamp or not last_activity or last_activity > cutoff_timestamp:
                    continue
                created_date = dt.datetime.fromtimestamp(created_timestamp, tz=dt.UTC).date()
                parser = ParagraphHTMLParser(paragraph_tags={"p", "li"})
                parser.feed(answer.get("body", ""))
                text = "\n\n".join(parser.paragraphs)
                answer_id = answer.get("answer_id")
                if not answer_id:
                    continue
                author = html.unescape(
                    answer.get("owner", {}).get("display_name", "Unknown user")
                )
                license_name, license_url = stackexchange_license(created_date)
                document = SourceDocument(
                    collection=collection,
                    document_id=f"stackoverflow-answer-{answer_id}",
                    title=f"Stack Overflow answer {answer_id}",
                    text=text,
                    source_url=f"https://stackoverflow.com/a/{answer_id}",
                    source_date=created_date.isoformat(),
                    authors=[author],
                    license_name=license_name,
                    license_url=license_url,
                    extra={
                        "answer_id": answer_id,
                        "question_id": answer.get("question_id"),
                        "score": answer.get("score"),
                        "last_activity_date": dt.datetime.fromtimestamp(
                            last_activity, tz=dt.UTC
                        ).date().isoformat(),
                        "attribution_url": f"https://stackoverflow.com/a/{answer_id}",
                    },
                )
                added = writer.append_document(document, target)
                if added:
                    print(
                        f"[stackexchange] {writer.source_count(collection)}/{target} "
                        f"from answer {answer_id}"
                    )

            if payload.get("quota_remaining", 1) <= 0:
                raise RuntimeError("Stack Exchange API quota exhausted; rerun after it resets")
            if not payload.get("has_more"):
                break

    if writer.source_count(collection) < target:
        raise RuntimeError(
            f"Stack Exchange returned only {writer.source_count(collection)}/{target} samples"
        )


GUTENBERG_START = re.compile(
    r"\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*{3}", re.I | re.S
)
GUTENBERG_END = re.compile(
    r"\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*{3}", re.I | re.S
)


def strip_gutenberg_boilerplate(text):
    start = GUTENBERG_START.search(text)
    end = GUTENBERG_END.search(text)
    if not start or not end or end.start() <= start.end():
        return None
    body = clean_inline_text(text[start.end() : end.start()])
    return strip_gutenberg_front_matter(body)


def strip_gutenberg_front_matter(text):
    """Remove a duplicated table-of-contents heading when it precedes the book."""
    heading_pattern = re.compile(
        r"(?im)^\s*((?:chapter|book|part)\s+(?:i|1|one)\b[^\n]{0,160})\s*$"
    )
    seen = {}
    matches = list(heading_pattern.finditer(text))
    for match in matches:
        normalized = re.sub(r"\W+", " ", match.group(1)).strip().lower()
        if normalized in seen:
            return clean_inline_text(text[match.start() :])
        seen[normalized] = match.start()
    return text


def parse_gutenberg_date(value):
    value = re.sub(r"\[[^]]*\]", "", value).strip()
    for pattern in ("%B %d, %Y", "%B, %Y", "%B %Y", "%Y-%m-%d", "%Y"):
        try:
            parsed = dt.datetime.strptime(value, pattern).date()
            if pattern in {"%B, %Y", "%B %Y"}:
                parsed = parsed.replace(day=calendar.monthrange(parsed.year, parsed.month)[1])
            if pattern == "%Y":
                parsed = parsed.replace(month=12, day=31)
            return parsed
        except ValueError:
            continue
    return None


def gutenberg_edition_dates(text):
    start = GUTENBERG_START.search(text)
    header = text[: start.start()] if start else text[:10_000]
    release_match = re.search(r"(?im)^\s*Release date:\s*(.+?)\s*$", header)
    updated_match = re.search(
        r"(?im)^\s*Most recently updated:\s*(.+?)\s*$", header
    )
    release_date = parse_gutenberg_date(release_match.group(1)) if release_match else None
    updated_date = parse_gutenberg_date(updated_match.group(1)) if updated_match else None
    return release_date, updated_date


def gutenberg_text_url(book):
    formats = book.get("formats", {})
    preferences = (
        "text/plain; charset=utf-8",
        "text/plain; charset=us-ascii",
        "text/plain",
    )
    for key in preferences:
        if formats.get(key):
            return formats[key]
    for key, value in formats.items():
        if key.startswith("text/plain") and value:
            return value
    return None


def collect_gutenberg(
    client, writer, target, cutoff
):
    collection = "gutenberg"
    if writer.source_count(collection) >= target:
        print(f"[{collection}] already has {writer.source_count(collection)} samples")
        return

    url = build_url(
        "https://gutendex.com/books/",
        {"languages": "en", "copyright": "false", "sort": "ascending"},
    )
    while url and writer.source_count(collection) < target:
        payload = client.json(url)
        for book in payload.get("results", []):
            if writer.source_count(collection) >= target:
                break
            if book.get("copyright") is not False:
                continue
            text_url = gutenberg_text_url(book)
            book_id = book.get("id")
            if not text_url or not book_id:
                continue
            try:
                raw_text = client.bytes(text_url).decode("utf-8-sig", errors="replace")
            except (urllib.error.URLError, urllib.error.HTTPError) as error:
                print(f"[gutenberg] skipped {book_id}: {error}", file=sys.stderr)
                continue
            release_date, updated_date = gutenberg_edition_dates(raw_text)
            edition_date = updated_date or release_date
            if not edition_date or edition_date > cutoff:
                continue
            text = strip_gutenberg_boilerplate(raw_text)
            if not text:
                continue
            authors = [
                author.get("name", "").strip()
                for author in book.get("authors", [])
                if author.get("name")
            ]
            death_years = [
                author.get("death_year")
                for author in book.get("authors", [])
                if author.get("death_year")
            ]
            document = SourceDocument(
                collection=collection,
                document_id=f"gutenberg-{book_id}",
                title=clean_inline_text(book.get("title", f"Gutenberg book {book_id}")),
                text=text,
                source_url=f"https://www.gutenberg.org/ebooks/{book_id}",
                source_date=edition_date.isoformat(),
                authors=authors,
                license_name="Public domain in the United States",
                license_url="https://www.gutenberg.org/policy/license.html",
                extra={
                    "gutenberg_id": book_id,
                    "cutoff_basis": "Project Gutenberg public-domain status and edition date",
                    "gutenberg_release_date": release_date.isoformat() if release_date else None,
                    "gutenberg_last_updated_date": (
                        updated_date.isoformat() if updated_date else None
                    ),
                    "latest_listed_author_death_year": max(death_years, default=None),
                    "access_api": "Gutendex catalog with Project Gutenberg text files",
                },
            )
            added = writer.append_document(document, target)
            if added:
                print(
                    f"[gutenberg] {writer.source_count(collection)}/{target} "
                    f"from ebook {book_id}"
                )
        url = payload.get("next")

    if writer.source_count(collection) < target:
        raise RuntimeError(
            f"Project Gutenberg returned only {writer.source_count(collection)}/{target} samples"
        )


OPENSTAX_REPOSITORIES = (
    "osbooks-physics",
    "osbooks-statistics",
    "osbooks-introduction-business",
    "osbooks-life-liberty-and-pursuit-happiness",
    "osbooks-astronomy",
    "osbooks-biology-bundle",
    "osbooks-psychology",
    "osbooks-introduction-sociology",
    "osbooks-principles-economics-bundle",
    "osbooks-college-physics-bundle",
    "osbooks-chemistry-bundle",
    "osbooks-university-physics-bundle",
    "osbooks-anatomy-physiology",
    "osbooks-microbiology",
    "osbooks-business-law",
    "osbooks-us-history",
    "osbooks-writing-guide",
    "osbooks-principles-finance",
    "osbooks-principles-of-management-bundle",
    "osbooks-introduction-anthropology",
    "osbooks-introduction-political-science",
)


def openstax_license_is_cc_by(license_text):
    normalized = re.sub(r"\s+", " ", license_text).lower()
    disallowed = ("noncommercial", "no derivatives", "noderivatives", "by-nc", "by-nd")
    if any(value in normalized for value in disallowed):
        return False
    return (
        "creativecommons.org/licenses/by/4.0" in normalized
        or "creative commons attribution 4.0 international" in normalized
        or "creative commons attribution license" in normalized
    )


def extract_openstax_module(xml_payload):
    try:
        root = ET.fromstring(xml_payload)
    except ET.ParseError:
        return None
    title = "Untitled OpenStax module"
    for element in root.iter():
        if local_name(element.tag) == "title":
            candidate = clean_inline_text(" ".join(element.itertext()))
            if candidate:
                title = candidate
                break

    blocked = {"figure", "table", "equation", "exercise", "solution", "media"}
    paragraphs = []

    def walk(element):
        name = local_name(element.tag)
        if name in blocked:
            return
        if name == "para":
            value = clean_inline_text(" ".join(element.itertext()))
            if word_count(value) >= 5:
                paragraphs.append(value)
            return
        for child in element:
            walk(child)

    walk(root)
    text = "\n\n".join(paragraphs)
    if word_count(text) < min(DEFAULT_TARGET_WORDS):
        return None
    return title, text


def collect_openstax(
    client, writer, target, cutoff
):
    collection = "openstax"
    if writer.source_count(collection) >= target:
        print(f"[{collection}] already has {writer.source_count(collection)} samples")
        return

    github_headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        github_headers["Authorization"] = f"Bearer {token}"

    until = f"{cutoff.isoformat()}T23:59:59Z"
    for repository in OPENSTAX_REPOSITORIES:
        if writer.source_count(collection) >= target:
            break
        commit_url = build_url(
            f"https://api.github.com/repos/openstax/{repository}/commits",
            {"until": until, "per_page": 1},
        )
        try:
            commits = client.json(commit_url, headers=github_headers)
        except urllib.error.HTTPError as error:
            print(f"[openstax] skipped {repository}: {error}", file=sys.stderr)
            continue
        if not isinstance(commits, list) or not commits:
            continue
        commit = commits[0]
        sha = commit.get("sha")
        commit_date = (
            commit.get("commit", {}).get("committer", {}).get("date", "")[:10]
        )
        if not sha or not commit_date or parse_date(commit_date) > cutoff:
            continue

        archive_url = f"https://codeload.github.com/openstax/{repository}/tar.gz/{sha}"
        try:
            archive_data = client.bytes(archive_url)
            archive = tarfile.open(fileobj=io.BytesIO(archive_data), mode="r:gz")
        except (urllib.error.URLError, urllib.error.HTTPError, tarfile.TarError) as error:
            print(f"[openstax] skipped {repository}: {error}", file=sys.stderr)
            continue

        members = [member for member in archive.getmembers() if member.isfile()]
        license_members = [
            member for member in members if Path(member.name).name.lower() in {"license", "license.txt"}
        ]
        license_text = ""
        for member in license_members:
            extracted = archive.extractfile(member)
            if extracted:
                license_text += extracted.read(200_000).decode("utf-8", errors="replace")
        if not openstax_license_is_cc_by(license_text):
            print(f"[openstax] rejected {repository}: snapshot is not verified CC BY 4.0")
            archive.close()
            continue

        module_members = sorted(
            (
                member
                for member in members
                if Path(member.name).suffix.lower() in {".cnxml", ".xml"}
                and "/modules/" in member.name
                and member.size <= 2_000_000
            ),
            key=lambda member: member.name,
        )
        for member in module_members:
            if writer.source_count(collection) >= target:
                break
            extracted = archive.extractfile(member)
            if not extracted:
                continue
            module = extract_openstax_module(extracted.read())
            if not module:
                continue
            module_title, text = module
            relative_path = member.name.split("/", 1)[-1]
            document_id = f"openstax-{repository}-{relative_path}"
            document = SourceDocument(
                collection=collection,
                document_id=document_id,
                title=module_title,
                text=text,
                source_url=(
                    f"https://github.com/openstax/{repository}/blob/{sha}/{relative_path}"
                ),
                source_date=commit_date,
                authors=["OpenStax"],
                license_name="CC BY 4.0",
                license_url=CC_BY_URL,
                extra={
                    "repository": f"openstax/{repository}",
                    "repository_commit": sha,
                    "repository_path": relative_path,
                    "cutoff_basis": "Last repository commit on or before cutoff",
                },
            )
            added = writer.append_document(document, target)
            if added:
                print(
                    f"[openstax] {writer.source_count(collection)}/{target} "
                    f"from {repository}/{relative_path}"
                )
        archive.close()

    if writer.source_count(collection) < target:
        raise RuntimeError(
            f"Verified OpenStax snapshots yielded only "
            f"{writer.source_count(collection)}/{target} samples"
        )


ARXIV_SETS = ("cs", "stat", "math", "physics", "q-bio", "q-fin", "econ", "eess")
ARXIV_LICENSES = {
    "http://creativecommons.org/licenses/by/4.0/": ("CC BY 4.0", CC_BY_URL),
    "https://creativecommons.org/licenses/by/4.0/": ("CC BY 4.0", CC_BY_URL),
    "http://creativecommons.org/licenses/by-sa/4.0/": ("CC BY-SA 4.0", CC_BY_SA_URL),
    "https://creativecommons.org/licenses/by-sa/4.0/": ("CC BY-SA 4.0", CC_BY_SA_URL),
    "http://creativecommons.org/publicdomain/zero/1.0/": ("CC0 1.0", CC0_URL),
    "https://creativecommons.org/publicdomain/zero/1.0/": ("CC0 1.0", CC0_URL),
}


def child_text(element, name):
    for child in element:
        if local_name(child.tag) == name:
            value = clean_inline_text(" ".join(child.itertext()))
            return value or None
    return None


def parse_arxiv_oai(payload):
    root = ET.fromstring(payload)
    records = []
    token = None
    for element in root.iter():
        if local_name(element.tag) == "resumptionToken":
            token = clean_inline_text("".join(element.itertext())) or None
        if local_name(element.tag) != "record":
            continue
        metadata = next(
            (child for child in element if local_name(child.tag) == "metadata"), None
        )
        if metadata is None or not list(metadata):
            continue
        arxiv = list(metadata)[0]
        authors = []
        for author in arxiv.iter():
            if local_name(author.tag) != "author":
                continue
            keyname = child_text(author, "keyname") or ""
            forenames = child_text(author, "forenames") or ""
            full_name = clean_inline_text(f"{forenames} {keyname}")
            if full_name:
                authors.append(full_name)
        records.append(
            {
                "id": child_text(arxiv, "id"),
                "created": child_text(arxiv, "created"),
                "updated": child_text(arxiv, "updated"),
                "title": child_text(arxiv, "title"),
                "license": child_text(arxiv, "license"),
                "categories": child_text(arxiv, "categories"),
                "doi": child_text(arxiv, "doi"),
                "authors": authors,
            }
        )
    return records, token


def latex_to_text(latex):
    latex = re.sub(r"(?m)(?<!\\)%.*$", " ", latex)
    latex = re.sub(
        r"\\begin\{(?:figure\*?|table\*?|equation\*?|align\*?|thebibliography|lstlisting|verbatim)\}"
        r".*?"
        r"\\end\{(?:figure\*?|table\*?|equation\*?|align\*?|thebibliography|lstlisting|verbatim)\}",
        " ",
        latex,
        flags=re.S,
    )
    document_match = re.search(r"\\begin\{document\}(.*)\\end\{document\}", latex, re.S)
    if document_match:
        latex = document_match.group(1)
    latex = re.sub(r"\\(?:begin|end)\{[^{}]+\}", "\n\n", latex)
    latex = re.sub(r"\$\$.*?\$\$|\\\[.*?\\\]|\\\(.*?\\\)", " ", latex, flags=re.S)
    latex = re.sub(r"(?<!\\)\$.*?(?<!\\)\$", " ", latex, flags=re.S)
    latex = re.sub(r"\\(?:cite\w*|ref|eqref|label|url|href)\s*(?:\[[^]]*\])?\s*\{[^{}]*\}", " ", latex)
    for _ in range(4):
        latex = re.sub(
            r"\\(?:section|subsection|subsubsection|paragraph|textbf|textit|emph|textrm|texttt|caption)\*?"
            r"\s*(?:\[[^]]*\])?\s*\{([^{}]*)\}",
            r"\n\n\1\n\n",
            latex,
        )
        latex = re.sub(r"\\[a-zA-Z@]+\*?\s*(?:\[[^]]*\])?\s*\{([^{}]*)\}", r" \1 ", latex)
    latex = re.sub(r"\\[a-zA-Z@]+\*?(?:\[[^]]*\])?", " ", latex)
    latex = latex.replace("~", " ").replace("\\&", "&")
    latex = re.sub(r"[{}]", " ", latex)
    latex = re.sub(r"\\[,_;!%#]", " ", latex)
    return clean_inline_text(latex)


def arxiv_source_text(payload):
    if payload.startswith(b"%PDF"):
        return None
    archives = []
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:*") as archive:
            for member in archive.getmembers():
                if (
                    member.isfile()
                    and Path(member.name).suffix.lower() in {".tex", ".ltx"}
                    and member.size <= 3_000_000
                ):
                    extracted = archive.extractfile(member)
                    if extracted:
                        archives.append((member.name, extracted.read()))
    except tarfile.TarError:
        decompressed = payload
        if payload.startswith(b"\x1f\x8b"):
            try:
                decompressed = gzip.decompress(payload)
            except gzip.BadGzipFile:
                return None
        if decompressed.startswith(b"%PDF"):
            return None
        archives.append(("source.tex", decompressed))

    if not archives:
        return None
    archives.sort(key=lambda item: (b"\\begin{document}" not in item[1], item[0]))
    latex = "\n\n".join(
        content.decode("utf-8", errors="replace") for _, content in archives
    )
    text = latex_to_text(latex)
    return text if word_count(text) >= min(DEFAULT_TARGET_WORDS) else None


def arxiv_oai_url(
    *,
    set_name = None,
    start = None,
    end = None,
    token = None,
):
    base = "https://oaipmh.arxiv.org/oai"
    if token:
        return build_url(base, {"verb": "ListRecords", "resumptionToken": token})
    if not set_name or not start or not end:
        raise ValueError("set_name, start, and end are required without a token")
    return build_url(
        base,
        {
            "verb": "ListRecords",
            "metadataPrefix": "arXiv",
            "from": start.isoformat(),
            "until": end.isoformat(),
            "set": set_name,
        },
    )


def collect_arxiv(
    client,
    writer,
    target,
    cutoff,
    arxiv_delay,
):
    collection = "arxiv"
    if writer.source_count(collection) >= target:
        print(f"[{collection}] already has {writer.source_count(collection)} samples")
        return

    per_set_target = math.ceil((target - writer.source_count(collection)) / len(ARXIV_SETS))
    processed_ids = set()

    for set_name in ARXIV_SETS:
        added_for_set = 0
        for year in range(cutoff.year, 2006, -1):
            if added_for_set >= per_set_target or writer.source_count(collection) >= target:
                break
            start = dt.date(year, 1, 1)
            end = min(dt.date(year, 12, 31), cutoff)
            token = None

            while added_for_set < per_set_target and writer.source_count(collection) < target:
                url = arxiv_oai_url(
                    set_name=set_name,
                    start=start,
                    end=end,
                    token=token,
                )
                try:
                    records, token = parse_arxiv_oai(
                        client.bytes(url, extra_delay=arxiv_delay)
                    )
                except (ET.ParseError, urllib.error.URLError, urllib.error.HTTPError) as error:
                    print(f"[arxiv] metadata error for {set_name}/{year}: {error}", file=sys.stderr)
                    break

                for record in records:
                    if added_for_set >= per_set_target or writer.source_count(collection) >= target:
                        break
                    arxiv_id = record.get("id")
                    license_info = ARXIV_LICENSES.get(record.get("license", ""))
                    created = parse_date(record.get("created"))
                    updated = parse_date(record.get("updated")) or created
                    if (
                        not arxiv_id
                        or arxiv_id in processed_ids
                        or not license_info
                        or not created
                        or created > cutoff
                        or not updated
                        or updated > cutoff
                    ):
                        continue
                    processed_ids.add(arxiv_id)
                    source_url = f"https://export.arxiv.org/e-print/{urllib.parse.quote(arxiv_id)}"
                    try:
                        text = arxiv_source_text(
                            client.bytes(source_url, extra_delay=arxiv_delay)
                        )
                    except (urllib.error.URLError, urllib.error.HTTPError) as error:
                        print(f"[arxiv] skipped {arxiv_id}: {error}", file=sys.stderr)
                        continue
                    if not text:
                        continue
                    document = SourceDocument(
                        collection=collection,
                        document_id=f"arxiv-{arxiv_id}",
                        title=record.get("title") or f"arXiv {arxiv_id}",
                        text=text,
                        source_url=f"https://arxiv.org/abs/{arxiv_id}",
                        source_date=(updated or created).isoformat(),
                        authors=record.get("authors", []),
                        license_name=license_info[0],
                        license_url=license_info[1],
                        extra={
                            "arxiv_id": arxiv_id,
                            "created_date": created.isoformat(),
                            "last_revision_date": updated.isoformat(),
                            "categories": (record.get("categories") or "").split(),
                            "doi": record.get("doi"),
                            "cutoff_basis": "Latest arXiv revision is on or before cutoff",
                        },
                    )
                    added = writer.append_document(document, target)
                    added_for_set += added
                    if added:
                        print(
                            f"[arxiv] {writer.source_count(collection)}/{target} "
                            f"from {arxiv_id}"
                        )
                if not token:
                    break

    if writer.source_count(collection) < target:
        raise RuntimeError(
            f"Openly licensed arXiv records yielded only "
            f"{writer.source_count(collection)}/{target} samples"
        )


COLLECTORS = {
    "pmc": lambda client, writer, target, cutoff, args: collect_epmc_source(
        client, writer, target, cutoff, plos_only=False
    ),
    "plos": lambda client, writer, target, cutoff, args: collect_epmc_source(
        client, writer, target, cutoff, plos_only=True
    ),
    "wikimedia": lambda client, writer, target, cutoff, args: collect_wikimedia(
        client,
        writer,
        target,
        cutoff,
        args.wikipedia_min_revision_bytes,
        args.wikipedia_workers,
    ),
    "stackexchange": lambda client, writer, target, cutoff, args: collect_stackexchange(
        client, writer, target, cutoff
    ),
    "gutenberg": lambda client, writer, target, cutoff, args: collect_gutenberg(
        client, writer, target, cutoff
    ),
    "openstax": lambda client, writer, target, cutoff, args: collect_openstax(
        client, writer, target, cutoff
    ),
    "arxiv": lambda client, writer, target, cutoff, args: collect_arxiv(
        client, writer, target, cutoff, args.arxiv_delay
    ),
}


def parse_sources(value):
    if value.strip().lower() == "all":
        return list(ALL_SOURCES)
    sources = [source.strip().lower() for source in value.split(",") if source.strip()]
    invalid = sorted(set(sources) - set(ALL_SOURCES))
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unknown sources: {', '.join(invalid)}. Choose from {', '.join(ALL_SOURCES)}"
        )
    return sources


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Dataset directory containing meta.json and human/ (default: data)",
    )
    parser.add_argument(
        "--sources",
        type=parse_sources,
        default=list(ALL_SOURCES),
        help=f"Comma-separated sources or 'all' (default: all). Choices: {', '.join(ALL_SOURCES)}",
    )
    parser.add_argument(
        "--target-per-source",
        type=int,
        default=3500,
        help="Desired total samples for each selected source (default: 3500)",
    )
    parser.add_argument(
        "--cutoff-date",
        type=dt.date.fromisoformat,
        default=DEFAULT_CUTOFF,
        help="Latest permitted source date in YYYY-MM-DD form (default: 2022-12-31)",
    )
    parser.add_argument(
        "--max-chunks-per-document",
        type=int,
        default=12,
        help="Limit any source document's contribution (default: 12)",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.2,
        help="Minimum delay between ordinary HTTP requests in seconds (default: 0.2)",
    )
    parser.add_argument(
        "--arxiv-delay",
        type=float,
        default=3.0,
        help="Minimum delay for arXiv requests in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--wikipedia-min-revision-bytes",
        type=int,
        default=8000,
        help="Skip short historical Wikipedia revisions (default: 8000 bytes)",
    )
    parser.add_argument(
        "--wikipedia-workers",
        type=int,
        default=6,
        help="Concurrent historical Wikipedia fetches (default: 6)",
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="Descriptive HTTP User-Agent sent to source APIs",
    )
    return parser


def main():
    args = build_parser().parse_args()
    if args.target_per_source <= 0:
        raise SystemExit("--target-per-source must be positive")
    if args.max_chunks_per_document <= 0:
        raise SystemExit("--max-chunks-per-document must be positive")
    if args.cutoff_date > DEFAULT_CUTOFF:
        raise SystemExit(
            "This human corpus collector intentionally caps source material at 2022-12-31"
        )

    writer = DatasetWriter(
        data_dir=args.data_dir,
        cutoff=args.cutoff_date,
        max_chunks=args.max_chunks_per_document,
    )
    client = HttpClient(args.user_agent, args.request_delay)

    failures = []
    for source in args.sources:
        print(
            f"\nStarting {source}: {writer.source_count(source)}/"
            f"{args.target_per_source} existing samples"
        )
        try:
            COLLECTORS[source](
                client,
                writer,
                args.target_per_source,
                args.cutoff_date,
                args,
            )
        except Exception as error:  # Continue independent sources and report loudly at the end.
            failures.append((source, str(error)))
            print(f"[{source}] FAILED: {error}", file=sys.stderr)

    writer.save()
    print("\nFinal licensed-source counts")
    for source in args.sources:
        print(f"  {source}: {writer.source_count(source)}")

    if failures:
        print("\nOne or more collectors did not reach their target:", file=sys.stderr)
        for source, message in failures:
            print(f"  {source}: {message}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
