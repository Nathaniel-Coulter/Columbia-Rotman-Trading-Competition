def print_header():
    header = r"""
 __  __            _        _  _____ _      _    
|  \/  | __ _ _ __| | _____| ||_   _(_) ___| | __
| |\/| |/ _` | '__| |/ / _ \ __|| | | |/ __| |/ /
| |  | | (_| | |  |   <  __/ |_ | | | | (__|   < 
|_|  |_|\__,_|_|  |_|\_\___|\__||_| |_|\___|_|\_\

Data Download Script V1.1     ©2025 MarketTick.net

--------------------------------------------------

This script downloads all the data you've purchased from MarketTick. After downloading, the compressed files are automatically unpacked.

1. Run this script and specify the full url of your personal download site
   Example: https://markettick.net/mt_api/data-buy_links.php?apikey=ABCDEF123456abcdef
2. The files will be downloaded automatically and unpacked in the "Downloads" subfolder.
"""
    print(header)
# -------------------------------
# Config
# -------------------------------
OUTPUT_DIR = r".\futures_research\data\raw"
CUSTOM_7z_PATH = ""

#Start MAIN Script
import os
import re
import json
import requests
import subprocess
import platform
import shutil
import urllib3

# Disable InsecureRequestWarning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def find_7z_binary() -> str:
    """Find correct 7-Zip executable depending on OS."""
    if CUSTOM_7z_PATH:
        return CUSTOM_7z_PATH

    system = platform.system()

    if system == "Windows":
        # Try PATH
        for candidate in ["7z.exe", "7za.exe"]:
            path = shutil.which(candidate)
            if path:
                return path

        # Try default install paths
        default_paths = [
            os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "7-Zip", "7z.exe"),
            os.path.join(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"), "7-Zip", "7z.exe"),
        ]
        for path in default_paths:
            if os.path.isfile(path):
                return path

        print("âš ï¸  7-Zip not found in PATH or default locations. Extraction will be skipped.")
        return None

    else:
        for candidate in ["7z", "7za"]:
            path = shutil.which(candidate)
            if path:
                return path

        print("âš ï¸  7-Zip not found in PATH. Extraction will be skipped.")
        return None


def load_html(source: str) -> str:
    """Load HTML from local file or URL."""
    if source.lower().startswith("http"):
        print(f"ðŸŒ Fetching HTML from {source}")
        r = requests.get(source, headers={"User-Agent": "Mozilla/5.0"}, verify=False)
        r.raise_for_status()
        return r.text
    else:
        print(f"ðŸ“‚ Reading HTML file {source}")
        with open(source, "r", encoding="utf-8") as f:
            return f.read()


def extract_json(html: str) -> list:
    """Extract JSON array from <script id='downloads-data'>."""
    match = re.search(
        r'<script[^>]+id=["\']downloads-data["\'][^>]*>(.*?)</script>',
        html,
        re.DOTALL | re.IGNORECASE,
    )
    if not match:
        print(f"âŒ No downloads-data JSON block found in HTML!")
        exit()
    json_text = match.group(1).strip()
    return json.loads(json_text)


def choose_section(entries: list) -> list:
    """Ask user if all sections or one specific section should be downloaded."""
    if len(entries) == 0:
        print(f"âŒ No downloads found in HTML! Check API Key!")
        exit()
    sections = sorted({f"{e['type']}_{e['symbol']}_{e['level']}" for e in entries})

    print("\nAvailable sections:")
    for i, sec in enumerate(sections, 1):
        print(f"{i}. {sec}")

    choice = input("\nDownload (a)ll sections or choose number: ").strip().lower()

    if choice == "a":
        return entries
    elif choice.isdigit() and 1 <= int(choice) <= len(sections):
        selected = sections[int(choice) - 1]
        return [e for e in entries if f"{e['type']}_{e['symbol']}_{e['level']}" == selected]
    else:
        print("âŒ Invalid choice, aborting.")
        return []


def download_and_extract(entry: dict, seven_zip: str | None):
    """Download file if missing and extract with 7-Zip (if available)."""
    section = f"{entry['type']}_{entry['symbol']}_{entry['level']}"
    section_dir = os.path.join(OUTPUT_DIR, section)
    os.makedirs(section_dir, exist_ok=True)

    filename = entry["filename"]
    filepath = os.path.join(section_dir, filename)
    extract_dir = os.path.join(section_dir, os.path.splitext(filename)[0])

    # Case 1: file and extracted folder already exist -> skip
    if os.path.exists(filepath) and os.path.isdir(extract_dir):
        print(f"â­ï¸ Skipping {filename} (already downloaded & extracted)")
        return

    # Case 2: file exists but not extracted -> extract only
    if os.path.exists(filepath) and not os.path.isdir(extract_dir):
        if seven_zip:
            print(f"ðŸ“‚ Extracting existing {filename} ...")
            os.makedirs(extract_dir, exist_ok=True)
            run_7zip(seven_zip, filepath, extract_dir, entry['password'])
            print(f"âœ… Extracted to {extract_dir}")
        else:
            print(f"âš ï¸ Skipping extraction for {filename} (7-Zip not found)")
        return

    # Case 3: file missing -> download first
    print(f"ðŸ“¥ Downloading {filename} ...")
    r = requests.get(entry["url"], headers={"User-Agent": "Mozilla/5.0"}, stream=True, verify=False)
    r.raise_for_status()
	
    runcount = 0
    with open(filepath, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                if runcount == 0:
                    chunk_str = chunk.decode("utf-8", errors="ignore")
                    if chunk_str.startswith("<h1>"):
                        print(chunk_str[len("<h1>"):(-len("</h1>"))-1])
                        f.close()
                        if os.path.exists(filepath):
                            os.remove(filepath) 
                        return
                runcount = 1
                f.write(chunk)

    print(f"âœ… Download complete: {filepath}")

    # Extract if possible
    if seven_zip:
        print(f"ðŸ“‚ Extracting {filename} ...")
        os.makedirs(extract_dir, exist_ok=True)
        run_7zip(seven_zip, filepath, extract_dir, entry['password'])
        print(f"âœ… Extracted to {extract_dir}")
    else:
        print(f"âš ï¸ Skipping extraction for {filename} (7-Zip not found)")


def run_7zip(seven_zip, filepath, extract_dir, password):
    """Run 7-Zip extraction with suppressed output unless error occurs."""
    try:
        result = subprocess.run(
            [seven_zip, "x", f"-p{password}", filepath, f"-o{extract_dir}", "-y"],
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"âŒ Extraction failed for {filepath}")
        print("---- 7-Zip Output ----")
        print(e.stdout)
        print(e.stderr)
		

def main():
    print_header()
    source = input("Enter the URL of your personal MarketTick download site: ").strip()
    html = load_html(source)
    entries = extract_json(html)
    to_process = choose_section(entries)

    seven_zip = find_7z_binary()

    for entry in to_process:
        print(f"")
        download_and_extract(entry, seven_zip)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"")
        print(f"")
        print(f"ðŸ‘‹ Goodbye!")