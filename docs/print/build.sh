#!/bin/sh
# Render a markdown file to an A4 PDF in the CIA report style (pandoc + headless Chrome).
# usage: sh docs/print/build.sh IN.md OUT.pdf TITLE
set -e
d=$(cd "$(dirname "$0")" && pwd -W 2>/dev/null || pwd)
pandoc "$1" -f markdown-auto_identifiers+lists_without_preceding_blankline --template "$d/template.html" --css "$d/report.css" --resource-path=".;$(dirname "$1")" --embed-resources --standalone --metadata pagetitle="$3" -o "$d/_tmp.html"
"/c/Program Files/Google/Chrome/Application/chrome.exe" --headless=new --disable-gpu --no-pdf-header-footer --print-to-pdf="$(pwd -W)/$2" "file:///$d/_tmp.html"
rm "$d/_tmp.html"
