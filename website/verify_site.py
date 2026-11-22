"""Check published pages against project evidence and local asset references."""
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ('src', 'href') and value:
                self.links.append(value)


def main():
    errors = []
    for file in [ROOT / 'index.html', ROOT / 'research.html'] + sorted(ROOT.glob('experiment-*.html')):
        source = file.read_text(encoding='utf-8')
        parser = Links()
        parser.feed(source)
        for url in parser.links:
            parsed = urlsplit(url)
            if parsed.scheme in ('https', 'http', 'mailto', 'data') or not parsed.path:
                continue
            if parsed.scheme or not (file.parent / unquote(parsed.path)).is_file():
                errors.append(f'{file.name}: broken or nonportable reference {url}')
        if 'Report text extraction was unavailable' in source:
            errors.append(f'{file.name}: missing archival report text')
    for n in range(1, 7):
        page = (ROOT / f'experiment-{n}.html').read_text(encoding='utf-8')
        metrics = json.loads((ROOT / f'assets/research/exp{n}_metrics.json').read_text())
        if 'CURRENT EVALUATION ADDENDUM' not in page or 'Archival report:' not in page:
            errors.append(f'Experiment {n}: missing current/archival distinction')
        if 'current-research-report' not in page or f'Experiment {n} Research Report' not in page:
            errors.append(f'Experiment {n}: revised full report missing')
        for section in ('Sources of error and errors identified','Assumptions and their consequences','Reproducibility and artifact record'):
            if section not in page:errors.append(f'Experiment {n}: missing revised report section {section}')
        if n == 6:
            score = metrics['confusion_matrix']['overall_accuracy'] * 100
            if f'{score:.2f}%' not in page or 'superseded' not in page:
                errors.append('Experiment 6: corrected score or supersession notice missing')
        if page.find(f'assets/research/exp{n}_overview.png') < page.find('class="result-gallery"'):
            errors.append(f'Experiment {n}: plots must follow result gallery')
    research=(ROOT/'research.html').read_text(encoding='utf-8')
    repeated=json.loads((ROOT/'assets/research/saved_analysis_reproduction.json').read_text())
    if repeated['status']!='passed' or f'{repeated["images_per_run"]:,}' not in research:
        errors.append('Research page does not match saved-analysis reproduction evidence')
    audit=json.loads((ROOT/'assets/research/source_content_summary.json').read_text())
    for count in ('original_files','decoded_unique_images','duplicate_files_beyond_first'):
        if f'{audit[count]:,}' not in research:
            errors.append(f'Research page does not match content audit: {count}')
    ablation=json.loads((ROOT/'assets/research/ablation_audit.json').read_text())
    for detector in ('fixed','calibrated'):
        for record in ablation['summary'][detector].values():
            if f'{record["mean_recall"]*100:.1f}%' not in research:
                errors.append('Research page does not match ablation recall')
    if 'Current research_v5 splits' not in research or 'No detected crack; distance undefined' not in research:
        errors.append('Research split version or missing-detection policy absent')
    for version in ('research_v4','research_v5'):
        screen=json.loads((ROOT/f'assets/research/{version}_summary.json').read_text())
        if f'{screen["cross_split_candidate_pairs"]:,}' not in research:
            errors.append(f'Research page does not match similarity screen: {version}')
    qualified=json.loads((ROOT/'assets/research/strict_rc_geometry_audit.json').read_text())
    for count in qualified['qualified_split_counts'].values():
        if f'{count}' not in research:
            errors.append('Research page does not match qualified dataset counts')
    if 'source-quality' not in research or 'publisher checksums' not in research:
        errors.append('Source-quality policy or publisher-identity limitation absent')
    if (ROOT/'assets/research/PRIVATE_review_key.csv').exists():
        errors.append('Private reviewer key must not be published')
    result = {'pages_checked': 8, 'errors': errors, 'passed': not errors,
              'browser_visual_review': 'not_performed; browser tooling unavailable'}
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
