import re
import time
import json
import logging
from dataclasses import dataclass
from typing import Optional, Iterable
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# Optional: "web agent" browser automation (stronger for JS-heavy sites)
# pip install playwright
# playwright install
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except Exception:
    PLAYWRIGHT_AVAILABLE = False


# -----------------------------
# Models
# -----------------------------

@dataclass(frozen=True)
class JobSourceResult:
    company_name: str
    career_page_url: str
    open_position_url: str


@dataclass(frozen=True)
class CompanyInfo:
    company_name: str
    company_website_url: str


# -----------------------------
# Config
# -----------------------------

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

COMMON_CAREERS_PATH_HINTS = (
    "/careers",
    "/career",
    "/jobs",
    "/job",
    "/join",
    "/join-us",
    "/work-with-us",
    "/about/careers",
    "/company/careers",
)

# If you find these words in link text/href, it’s likely the careers page
CAREER_KEYWORDS = re.compile(r"\b(careers?|jobs?|join|openings|vacancies|work with us)\b", re.I)

# If a URL contains these, it’s likely a job posting link
JOB_POSTING_HINTS = re.compile(r"\b(job|jobs|careers|position|opening|vacancy|req|requisition)\b", re.I)


# -----------------------------
# Utilities
# -----------------------------

def normalize_url(url: str) -> str:
    url = url.strip()
    if not url:
        return url
    # Add scheme if missing
    parsed = urlparse(url)
    if not parsed.scheme:
        url = "https://" + url.lstrip("/")
    return url


def same_domain(a: str, b: str) -> bool:
    return urlparse(a).netloc.lower() == urlparse(b).netloc.lower()


def fetch_html(url: str, timeout_s: int = 20) -> str:
    r = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout_s, allow_redirects=True)
    r.raise_for_status()
    return r.text


def extract_links(html: str, base_url: str) -> list[tuple[str, str]]:
    """Return list of (absolute_url, link_text)."""
    soup = BeautifulSoup(html, "html.parser")
    links: list[tuple[str, str]] = []
    for a in soup.select("a[href]"):
        href = a.get("href") or ""
        text = " ".join(a.get_text(" ", strip=True).split())
        abs_url = urljoin(base_url, href)
        # Skip mailto/tel/javascript
        if abs_url.startswith(("mailto:", "tel:", "javascript:")):
            continue
        links.append((abs_url, text))
    return links


def pick_best_career_link(links: Iterable[tuple[str, str]], company_home: str) -> Optional[str]:
    """
    Pick best career page candidate from links on the homepage.
    Preference order:
      1) same-domain links with career keywords
      2) any link with career keywords
      3) heuristic common paths appended to base
    """
    same_domain_hits = []
    other_hits = []

    for url, text in links:
        if CAREER_KEYWORDS.search(url) or CAREER_KEYWORDS.search(text):
            if same_domain(url, company_home):
                same_domain_hits.append(url)
            else:
                other_hits.append(url)

    if same_domain_hits:
        return same_domain_hits[0]
    if other_hits:
        return other_hits[0]
    return None


def fallback_guess_career_page(company_home: str) -> Optional[str]:
    """Try common careers paths."""
    company_home = normalize_url(company_home)
    for path in COMMON_CAREERS_PATH_HINTS:
        candidate = urljoin(company_home.rstrip("/") + "/", path.lstrip("/"))
        try:
            r = requests.head(candidate, headers=DEFAULT_HEADERS, timeout=10, allow_redirects=True)
            if 200 <= r.status_code < 400:
                return r.url  # final URL after redirects
        except Exception:
            continue
    return None


def pick_one_job_posting_link(links: Iterable[tuple[str, str]], career_page_url: str) -> Optional[str]:
    """
    Pick one opening position link from the career page.
    Heuristic:
      - prefer same-domain
      - prefer URLs containing job hints
      - avoid navigation links (privacy, terms, etc.)
    """
    bad_words = re.compile(r"\b(privacy|terms|cookies|linkedin|facebook|twitter|instagram)\b", re.I)

    candidates = []
    for url, text in links:
        if bad_words.search(url) or bad_words.search(text):
            continue
        if JOB_POSTING_HINTS.search(url) or JOB_POSTING_HINTS.search(text):
            score = 0
            if same_domain(url, career_page_url):
                score += 2
            if "apply" in url.lower() or "apply" in text.lower():
                score += 1
            candidates.append((score, url))

    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


# -----------------------------
# LinkedIn extraction (two options)
# -----------------------------

def get_company_info_from_linkedin_via_api(linkedin_job_url: str) -> CompanyInfo:
    """
    Template for using a third-party LinkedIn crawler API (recommended).
    Replace with your provider call.

    Expected output: company name + company website URL.
    """
    # EXAMPLE STUB (pseudo):
    # api_key = os.environ["YOUR_PROVIDER_KEY"]
    # resp = requests.get(
    #   "https://api.provider.com/linkedin/job",
    #   params={"url": linkedin_job_url},
    #   headers={"Authorization": f"Bearer {api_key}"},
    #   timeout=30
    # )
    # resp.raise_for_status()
    # data = resp.json()
    # return CompanyInfo(company_name=data["company"]["name"],
    #                    company_website_url=data["company"]["website"])

    raise NotImplementedError("Plug in your LinkedIn provider API here.")


def get_company_info_from_linkedin_best_effort_html(linkedin_job_url: str) -> CompanyInfo:
    """
    Best-effort parsing of LinkedIn job page HTML.
    Often fails due to auth, dynamic content, bot protections.
    Use only as fallback.
    """
    html = fetch_html(linkedin_job_url)
    soup = BeautifulSoup(html, "html.parser")

    # Heuristic: company name often appears in title/meta
    title = (soup.title.get_text(strip=True) if soup.title else "") or ""
    # Example: "Software Engineer at CompanyName | LinkedIn"
    m = re.search(r"\bat\s+(.+?)\s*\|\s*LinkedIn\b", title, re.I)
    company_name = m.group(1).strip() if m else "Unknown Company"

    # Website URL is rarely present in job page HTML. You may only get company profile URL.
    # We'll try a couple guesses, but often you'll need the API.
    company_website_url = ""
    for a in soup.select("a[href]"):
        href = a.get("href") or ""
        # If any external link looks like a company homepage, grab it.
        if href.startswith("http") and "linkedin.com" not in href:
            company_website_url = href
            break

    if not company_website_url:
        raise RuntimeError("Could not extract company website from LinkedIn HTML. Use a 3rd-party API.")

    return CompanyInfo(company_name=company_name, company_website_url=company_website_url)


# -----------------------------
# Web agent to find career page + job opening
# -----------------------------

def find_career_page_url(company_website_url: str, use_playwright: bool = False) -> str:
    company_website_url = normalize_url(company_website_url)

    # Option A: requests + bs4 (fast, cheap)
    if not use_playwright:
        home_html = fetch_html(company_website_url)
        links = extract_links(home_html, company_website_url)

        career = pick_best_career_link(links, company_website_url)
        if career:
            return career

        guessed = fallback_guess_career_page(company_website_url)
        if guessed:
            return guessed

        raise RuntimeError("Could not find career page from homepage heuristics.")

    # Option B: Playwright "web agent" (handles JS menus, dynamic content)
    if not PLAYWRIGHT_AVAILABLE:
        raise RuntimeError("Playwright not installed/available. pip install playwright && playwright install")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent=DEFAULT_HEADERS["User-Agent"])
        page.goto(company_website_url, wait_until="domcontentloaded", timeout=45_000)

        # Collect visible links
        anchors = page.query_selector_all("a[href]")
        candidates = []
        for a in anchors:
            href = a.get_attribute("href") or ""
            text = (a.inner_text() or "").strip()
            if CAREER_KEYWORDS.search(href) or CAREER_KEYWORDS.search(text):
                candidates.append(urljoin(company_website_url, href))

        browser.close()

        if candidates:
            return candidates[0]

    guessed = fallback_guess_career_page(company_website_url)
    if guessed:
        return guessed
    raise RuntimeError("Could not find career page (Playwright + fallback).")


def find_one_open_position_url(career_page_url: str, use_playwright: bool = False) -> str:
    career_page_url = normalize_url(career_page_url)

    if not use_playwright:
        html = fetch_html(career_page_url)
        links = extract_links(html, career_page_url)
        job_url = pick_one_job_posting_link(links, career_page_url)
        if job_url:
            return job_url
        raise RuntimeError("Could not find a job posting link on the career page via HTML parsing.")

    if not PLAYWRIGHT_AVAILABLE:
        raise RuntimeError("Playwright not installed/available. pip install playwright && playwright install")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent=DEFAULT_HEADERS["User-Agent"])
        page.goto(career_page_url, wait_until="domcontentloaded", timeout=45_000)

        anchors = page.query_selector_all("a[href]")
        candidates = []
        for a in anchors:
            href = a.get_attribute("href") or ""
            text = (a.inner_text() or "").strip()
            abs_url = urljoin(career_page_url, href)

            if JOB_POSTING_HINTS.search(abs_url) or JOB_POSTING_HINTS.search(text):
                candidates.append(abs_url)

        browser.close()

        if candidates:
            return candidates[0]

    raise RuntimeError("Could not find a job posting link on the career page (Playwright).")


# -----------------------------
# Orchestrator (the "agent")
# -----------------------------

def run_ai_job_source_agent(
    linkedin_job_url: str,
    *,
    prefer_linkedin_api: bool = True,
    use_playwright_for_web_agent: bool = False,
    polite_delay_s: float = 0.8,
) -> JobSourceResult:
    """
    Main pipeline. Returns:
      company_name, career_page_url, open_position_url
    """
    logging.info("Starting agent for LinkedIn job URL: %s", linkedin_job_url)

    # 1) Company name + website
    if prefer_linkedin_api:
        try:
            company = get_company_info_from_linkedin_via_api(linkedin_job_url)
        except NotImplementedError:
            # If you didn't plug in API yet, fallback
            company = get_company_info_from_linkedin_best_effort_html(linkedin_job_url)
    else:
        company = get_company_info_from_linkedin_best_effort_html(linkedin_job_url)

    time.sleep(polite_delay_s)

    # 2) Find career page
    career_page_url = find_career_page_url(company.company_website_url, use_playwright=use_playwright_for_web_agent)

    time.sleep(polite_delay_s)

    # 3) Find one opening position URL
    open_position_url = find_one_open_position_url(career_page_url, use_playwright=use_playwright_for_web_agent)

    return JobSourceResult(
        company_name=company.company_name,
        career_page_url=career_page_url,
        open_position_url=open_position_url,
    )


# -----------------------------
# Example usage
# -----------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    linkedin_url = "https://www.linkedin.com/jobs/view/<JOB_ID>/"

    try:
        result = run_ai_job_source_agent(
            linkedin_url,
            prefer_linkedin_api=True,              # recommended
            use_playwright_for_web_agent=False,    # set True if websites are JS-heavy
        )
        # Output format requested:
        print(result.company_name, result.career_page_url, result.open_position_url, sep=",")
    except Exception as e:
        logging.exception("Agent failed: %s", e)
