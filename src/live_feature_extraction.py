import re
import requests
import pandas as pd

from bs4 import BeautifulSoup
from urllib.parse import urlparse


# -------------------------------------------------
# Feature Extraction Function
# -------------------------------------------------

def extract_features(url):

    headers = {
        "User-Agent":
        "Mozilla/5.0"
    }

    try:

        response = requests.get(
            url,
            timeout=10,
            headers=headers
        )

        html = response.text

        soup = BeautifulSoup(html, "lxml")

    except Exception:

        soup = BeautifulSoup("", "lxml")

        html = ""

    # -----------------------------------------
    # URL Information
    # -----------------------------------------

    parsed = urlparse(url)

    domain = parsed.netloc

    url_length = len(url)

    domain_length = len(domain)

    is_https = int(url.startswith("https"))

    has_https = is_https

    dot_count = url.count(".")

    slash_count = url.count("/")

    hyphen_count = url.count("-")

    digit_count = sum(c.isdigit() for c in url)

    letter_count = sum(c.isalpha() for c in url)

    special_char_count = len(
        re.findall(r"[^a-zA-Z0-9]", url)
    )

    ip_pattern = r"(\d{1,3}\.){3}\d{1,3}"

    is_domain_ip = int(
        bool(re.search(ip_pattern, domain))
    )
    # -----------------------------------------
    # HTML Features
    # -----------------------------------------

    title = soup.title.string.strip() if soup.title and soup.title.string else ""

    has_title = int(len(title) > 0)

    meta_description = soup.find("meta", attrs={"name": "description"})

    has_description = int(meta_description is not None)

    favicon = soup.find("link", rel=lambda x: x and "icon" in x.lower())

    has_favicon = int(favicon is not None)

    images = soup.find_all("img")

    no_of_image = len(images)

    css_files = soup.find_all("link", rel="stylesheet")

    no_of_css = len(css_files)

    js_files = soup.find_all("script")

    no_of_js = len(js_files)

    forms = soup.find_all("form")

    submit_buttons = soup.find_all(
        "input",
        {"type": "submit"}
    )

    has_submit_button = int(
        len(forms) > 0 or len(submit_buttons) > 0
    )

    hidden_fields = soup.find_all(
        "input",
        {"type": "hidden"}
    )

    has_hidden_fields = int(
        len(hidden_fields) > 0
    )

    password_fields = soup.find_all(
        "input",
        {"type": "password"}
    )

    has_password_field = int(
        len(password_fields) > 0
    )

    social_keywords = [
        "facebook",
        "twitter",
        "instagram",
        "linkedin",
        "youtube"
    ]

    has_social_net = int(
        any(keyword in html.lower() for keyword in social_keywords)
    )

    has_copyright = int(
        "copyright" in html.lower() or "©" in html
    )

    responsive = soup.find(
        "meta",
        attrs={"name": "viewport"}
    )

    is_responsive = int(
        responsive is not None
    )

    robots = int(
        soup.find("meta", attrs={"name": "robots"}) is not None
    )

    links = soup.find_all("a", href=True)

    no_of_external_ref = sum(
        1
        for link in links
        if link["href"].startswith("http")
    )
    # -----------------------------------------
    # Create Feature Dictionary
    # -----------------------------------------

    features = {

        "URLLength": url_length,
        "DomainLength": domain_length,
        "IsDomainIP": is_domain_ip,
        "URLSimilarityIndex": 0,
        "CharContinuationRate": 0,
        "TLDLegitimateProb": 0,
        "URLCharProb": 0,
        "NoOfSubDomain": domain.count("."),
        "HasObfuscation": int("%" in url or "@" in url),
        "NoOfObfuscatedChar": url.count("%") + url.count("@"),
        "NoOfLettersInURL": letter_count,
        "NoOfDegitsInURL": digit_count,
        "NoOfOtherSpecialCharsInURL": special_char_count,
        "IsHTTPS": is_https,
        "HasTitle": has_title,
        "DomainTitleMatchScore": 0,
        "URLTitleMatchScore": 0,
        "HasFavicon": has_favicon,
        "Robots": robots,
        "IsResponsive": is_responsive,
        "NoOfURLRedirect": 0,
        "HasDescription": has_description,
        "HasSocialNet": has_social_net,
        "HasSubmitButton": has_submit_button,
        "HasHiddenFields": has_hidden_fields,
        "HasPasswordField": has_password_field,
        "Bank": int("bank" in url.lower()),
        "Pay": int("pay" in url.lower()),
        "Crypto": int("crypto" in url.lower()),
        "HasCopyrightInfo": has_copyright,
        "NoOfImage": no_of_image,
        "NoOfCSS": no_of_css,
        "NoOfJS": no_of_js,
        "NoOfExternalRef": no_of_external_ref,
        "URL_Length": url_length,
        "Dot_Count": dot_count,
        "Slash_Count": slash_count,
        "Hyphen_Count": hyphen_count,
        "Digit_Count": digit_count,
        "HasHTTPS": has_https

    }

    return pd.DataFrame([features])