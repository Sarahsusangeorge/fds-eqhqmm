"""Download the reference PDFs listed in references.csv.  Run on your own computer (needs internet):

    pip install requests
    python fetch_references.py --email you@example.com

Tries the direct link first, then Unpaywall (free open-access copies, looked up by DOI). Papers behind a paywall
(for example Econometrica, IEEE, some Elsevier) cannot be fetched this way: they are listed in MANUAL_DOWNLOAD.txt;
download those through your university library login and save them as <key>.pdf in this folder.
"""
import argparse, csv, pathlib, sys
import requests

H = {"User-Agent": "Mozilla/5.0 (research reference fetch)"}


def get_pdf(url):
    r = requests.get(url, headers=H, timeout=60, allow_redirects=True)
    if r.ok and r.content[:4] == b"%PDF":
        return r.content
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True, help="used only for the Unpaywall API (they ask for one)")
    a = ap.parse_args()
    here = pathlib.Path(__file__).parent
    manual = []
    for row in csv.DictReader(open(here / "references.csv", encoding="utf-8")):
        out = here / f"{row['key']}.pdf"
        if out.exists():
            continue
        data = None
        try:
            if row["direct_url"]:
                data = get_pdf(row["direct_url"])
            if data is None and row["doi"]:
                j = requests.get(f"https://api.unpaywall.org/v2/{row['doi']}", params={"email": a.email}, timeout=30).json()
                loc = j.get("best_oa_location") or {}
                for u in (loc.get("url_for_pdf"), loc.get("url")):
                    if u:
                        data = get_pdf(u)
                        if data:
                            break
        except Exception as e:  # keep going
            print("error", row["key"], e)
        if data:
            out.write_bytes(data)
            print("saved ", out.name)
        else:
            manual.append(f"{row['key']}: {row['citation']}  DOI {row['doi'] or '-'}")
            print("MANUAL", row["key"])
    (here / "MANUAL_DOWNLOAD.txt").write_text("\n".join(manual) + "\n", encoding="utf-8")
    print(f"{len(manual)} need manual download (see MANUAL_DOWNLOAD.txt)")
    # the base paper also goes into 2_Base_Paper
    song = here / "Song2025.pdf"
    if song.exists():
        dest = here.parent / "2_Base_Paper" / "Song_2025_HQMM_stock_prediction.pdf"
        dest.write_bytes(song.read_bytes())
        print("base paper copied to", dest)


if __name__ == "__main__":
    sys.exit(main())
