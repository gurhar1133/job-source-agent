import re
import time
import os
import json
import logging
import pandas as pd
import requests
from dataclasses import dataclass
from typing import Optional, Iterable
from urllib.parse import urljoin, urlparse, urlunparse
from playwright.sync_api import sync_playwright
from serpapi import GoogleSearch
from openai import OpenAI
from bs4 import BeautifulSoup


# GOAL: AI Job Source agent
# 1) From Linkedin job listing pages, crawl the company name and company website URL(You may utilize third party Linkedin crawler API if you need)
# 2) From company website URL,use web agent to navigate to the career page URL
# 3) From the career page, get one opening position's URL
# 4) Return the results in format: company name,career page URL,open position's URL


# TODO: read from config
_SERPER_API_KEY = os.environ["SERPER_API_KEY"]
_OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
_EXAMPLES = [
    "4365358530",
    "4342555572",
    "4306115994",
    "4326760101",
    "4361769350",
    "4364972673",
    "4115499029",
    "4367451488",
    "4342953214",
    "4332433653"

]
_BAD_DOMAINS = [
    "linkedin.com",
    "wikipedia.org",
    "glassdoor.com",
    "indeed.com",
    "crunchbase.com",
    "pitchbook.com",
    "zoominfo.com",
    "facebook.com",
    "twitter.com",
]
_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
_COMMON_CAREERS_PATH_HINTS = (
    "/careers",
    "/career",
    "/jobs",
    "/job",
    "/join",
    "/join-us",
    "/work-with-us",
    "/about/careers",
    "/company/careers",
    "/open-roles",
)
_CAREER_KEYWORDS = re.compile(r"\b(careers?|jobs?|join|openings|vacancies|work with us)\b", re.I)


@dataclass(frozen=True)
class JobSourceResult:
    job_title: str
    company_name: str
    career_page_url: str
    open_position_url: str


@dataclass(frozen=True)
class CompanyInfo:
    company_name: str
    company_website_url: str


def is_valid_company_domain(url: str, company_name: str) -> bool:
    domain = urlparse(url).netloc.lower()
    if any(bad in domain for bad in _BAD_DOMAINS):
        return False
    tokens = company_name.lower().replace(",", "").split()
    return any(token in domain for token in tokens)


def search_company_website(company_name):
    params = {
        "engine": "google",
        "q":  f"{company_name} official website",
        "google_domain": "google.com",
        "hl": "en",
        "gl": "us",
        "api_key": _SERPER_API_KEY
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    organic = results.get("organic_results", [])
    for result in organic:
        link = result.get("link")
        if not link:
            continue

        if is_valid_company_domain(link, company_name):
            return clean_root(link)
    return None


def normalize_url(url):
    url = url.strip()
    if not url:
        return url
    parsed = urlparse(url)
    if not parsed.scheme:
        url = "https://" + url.lstrip("/")
    return url


def same_domain(a, b):
    return urlparse(a).netloc.lower() == urlparse(b).netloc.lower()


def fetch_html(url, timeout_s=20):
    r = requests.get(url, headers=_DEFAULT_HEADERS, timeout=timeout_s, allow_redirects=True)
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
        if _CAREER_KEYWORDS.search(url) or _CAREER_KEYWORDS.search(text):
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
    print("company home", company_home)
    for path in _COMMON_CAREERS_PATH_HINTS:
        candidate = urljoin(company_home.rstrip("/") + "/", path.lstrip("/"))
        try:
            r = requests.head(candidate, headers=_DEFAULT_HEADERS, timeout=10, allow_redirects=True)
            if 200 <= r.status_code < 400:
                return r.url
        except Exception:
            continue

        try:
            r = requests.head("careers." + company_home, headers=_DEFAULT_HEADERS, timeout=10, allow_redirects=True)
            if 200 <= r.status_code < 400:
                return r.url
        except Exception:
            continue

    return None


def extract_company_from_title(title: str) -> str:
    if " hiring " in title:
        return title.split(" hiring ")[0].strip()
    return None


def clean_root(url: str) -> str:
    p = urlparse(url)
    return urlunparse((p.scheme, p.netloc, "", "", "", ""))


def get_company_info_from_linkedin_best_effort_html(linkedin_job_url):
    html = fetch_html(linkedin_job_url)
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.get_text(strip=True) if soup.title else "") or ""
    company_name = extract_company_from_title(title)
    print("company name", company_name)
    company_website_url = search_company_website(company_name)
    print("company website", company_website_url)
    return CompanyInfo(company_name=company_name, company_website_url=company_website_url)


def find_career_page_url(company_website_url):
    company_website_url = normalize_url(company_website_url)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent=_DEFAULT_HEADERS["User-Agent"])
        try:
            page.goto(company_website_url, wait_until="domcontentloaded", timeout=45_000)

            # Collect visible links
            anchors = page.query_selector_all("a[href]")
            candidates = []
            for a in anchors:
                href = a.get_attribute("href") or ""
                text = (a.inner_text() or "").strip()
                if _CAREER_KEYWORDS.search(href) or _CAREER_KEYWORDS.search(text):
                    candidates.append(urljoin(company_website_url, href))
            browser.close()
            if candidates:
                return candidates[0]
        except Exception:
            print("playwright failure")
    guessed = fallback_guess_career_page(company_website_url)
    if guessed:
        return guessed
    raise RuntimeError("Could not find career page (Playwright + fallback).")


def fetch_page_html(url):
    html = ""
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, timeout=60_000)  # up to 60 seconds for navigation
        try:
            page.wait_for_load_state("networkidle")
        except Exception:
            print("faild to wait for network idle")
        # page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        # page.wait_for_timeout(1500)

        # Check frames first
        # for frame in page.frames:
        #     html += frame.content()
        html += page.content()
        browser.close()
    return html


def open_positions_agent(url):
    print("jobs_site:", url)
    client = OpenAI(api_key=_OPENAI_API_KEY)

    def fetch_url(target_url):
        return fetch_page_html(target_url)
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "fetch_url",
                "description": "Fetch raw HTML from a URL (used when jobs are embedded via iframe/ATS).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "Absolute URL to fetch"},
                    },
                    "required": ["url"],
                },
            },
        }
    ]

    html = fetch_page_html(url)
    # print(html)
    # default = str({
    #     "title": "No specific roles detected (check careers url)",
    #     "url": url,
    # })
    messages = [
        {
            "role": "system",
            "content": (
                "You are a web-crawling assistant. You read through web pages in order to find job postings.\n\n"

                "GOAL\n"
                "Return EXACTLY ONE job posting: a specific job title + a DIRECT URL to the job detail page.\n\n"

                "WHAT COUNTS AS A REAL JOB POSTING\n"
                "- A specific role title (e.g., 'Senior Software Engineer', 'Data Scientist II').\n"
                "- The URL must open a job detail page.\n"
                "- The job detail page should include at least ONE of:\n"
                "  (a) responsibilities/duties\n"
                "  (b) qualifications/requirements\n"
                "  (c) an 'Apply' button/link\n"
                "  (d) job description text\n\n"

                "WHAT DOES NOT COUNT\n"
                "- Careers home pages.\n"
                "- Job search/listing pages with many roles.\n"
                "- Category/department pages (Engineering, Sales, etc.) unless they already show a single job detail.\n"
                "- Links/buttons like: 'Open Roles', 'Explore Roles', 'View Opportunities', 'Search Jobs', 'Job Postings'.\n\n"

                "DETERMINISTIC SELECTION RULE (when multiple real jobs exist)\n"
                "Choose the FIRST valid job detail link in DOM order (top-to-bottom, left-to-right).\n"
                "If multiple links point to the same job, choose the shortest canonical URL.\n\n"

                "NAVIGATION / TOOL RULES (use fetch_url)\n"
                "You may call fetch_url ONLY when a real job detail page is NOT already present in the current HTML.\n"
                "Call fetch_url for EXACTLY ONE target per step.\n\n"
                "Fetch priority (first match wins):\n"
                "1) If an iframe likely contains jobs or an ATS embed exists, fetch the iframe src (absolute URL).\n"
                "2) Else, if there is an external ATS link/domain (Greenhouse, Lever, Workday, Ashby, SmartRecruiters, iCIMS, Taleo, BambooHR),\n"
                "   fetch the most job-like URL (absolute) pointing there.\n"
                "3) Else, fetch the most job-like internal link (absolute) whose href/text suggests jobs, e.g. contains:\n"
                "   /jobs, /careers, /join-us, /open-roles, /positions, /opportunities.\n"
                "4) Else, if only department/section links exist (Engineering, Sales, HR, etc.), fetch the FIRST such subsection link (absolute).\n"
                "- Prefer URLs that look like a specific posting, e.g. contain: /job/, /jobs/, /positions/, 'gh_jid=', 'lever.co/', 'workday', 'ashby', 'icims'.\n"
                "- Avoid pure filters/search pages unless they are the only path to reach a detail page.\n"
                "- When in doubt, call fetch_url on the best available guess for a job post\n\n"

                # "FINAL SELF-CHECK (must be true before you answer)\n"
                # "- Title is a specific role (not generic CTA).\n"
                # "- URL is a direct job detail page.\n"
                # "- Output is valid JSON only.\n\n"

                "OUTPUT\n"
                "Return ONLY JSON:\n"
                "- If found: {\"title\": \"...\", \"url\": \"...\"}\n"
                "- If not found, make an educated guess based on the evidence. Indicate uncertainty in your response by adding a question mark to the job title.\n"
            )

        },
        {
            "role": "user",
            "content": (
                f"Base URL: {url}\n\n"
                f"HTML:\n{html}"
            ),
        },
    ]

    for _ in range(3):
        resp = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.0,
            max_tokens=500,
        )

        msg = resp.choices[0].message
        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                if tc.function.name == "fetch_url":
                    args = json.loads(tc.function.arguments)
                    # Make iframe URLs absolute if needed
                    target = args["url"]
                    target = urljoin(url, target)
                    print(f"Looks like the jobs might be posted at {target}. Navigating there...")
                    fetched_html = fetch_url(target)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": fetched_html,
                    })
                    
            continue
        return msg.content.strip()
    return '{"no_jobs_found": true, "reason": "Exceeded tool hops"}'


def run_ai_job_source_agent(
    linkedin_job_url,
    *,
    polite_delay_s=0.01,
):
    """
    Main pipeline. Returns:
      company_name, career_page_url, open_position_url
    """
    logging.info("Starting agent for LinkedIn job URL: %s", linkedin_job_url)

    # 1) Company name + website
    
    company = get_company_info_from_linkedin_best_effort_html(linkedin_job_url)

    time.sleep(polite_delay_s)

    # 2) Find career page
    career_page_url = find_career_page_url(company.company_website_url)

    time.sleep(polite_delay_s)

    # 3) Find one opening position URL
    res = json.loads(open_positions_agent(career_page_url))
    print('res:', res)
    open_position_url = res["url"]
    job_title = res["title"]

    return JobSourceResult(
        job_title=job_title,
        company_name=company.company_name,
        career_page_url=career_page_url,
        open_position_url=open_position_url,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    final_results = []
    for job_id in _EXAMPLES:
        linkedin_url = f"https://www.linkedin.com/jobs/view/{job_id}/"
        try:
            result = run_ai_job_source_agent(linkedin_url)
            print("-"*80)
            print("\n\n")
            print("RESULTS:")
            print("Job title:", result.job_title )
            print("INFO:")
            print(f"{result.company_name}, {result.career_page_url}, {result.open_position_url}")
            final_results.append({
                "company": result.company_name,
                "career_page": result.career_page_url,
                "job title": result.job_title,
                "job link": result.open_position_url
            })
            print("\n\n")
            print("-"*80)
        except Exception as e:
            logging.exception("Agent failed: %s", e)
    df = pd.DataFrame(final_results)
    print(df)
    df.to_csv("jobs.csv")
