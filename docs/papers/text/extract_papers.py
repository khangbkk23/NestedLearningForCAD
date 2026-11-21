import hashlib, json, sys, datetime, pathlib, pypdf
root = pathlib.Path(sys.argv[1]); out = root / 'text'; out.mkdir(exist_ok=True)
man = []
for pdf in sorted(list(root.glob('*.pdf')) + list((root / 'related_cad').glob('*.pdf'))):
    sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
    r = pypdf.PdfReader(str(pdf))
    pages = [(p.extract_text() or '') for p in r.pages]
    body = ''.join(f'\n\n===== PAGE {i+1} =====\n{t}' for i, t in enumerate(pages))
    rel = pdf.relative_to(root)
    dst = out / (str(rel.with_suffix('.txt')).replace('/', '__'))
    hdr = (f'# SOURCE: docs/papers/{rel}\n# SOURCE_SHA256: {sha}\n# EXTRACTOR: pypdf {pypdf.__version__} '
           f'(PdfReader.extract_text, per page)\n# EXTRACTED: {datetime.date.today()}\n'
           '# NOTE: machine extraction; equations/tables/unicode may be garbled. Cite the PDF, not this file.\n')
    dst.write_text(hdr + body, encoding='utf-8')
    man.append({'pdf': f'docs/papers/{rel}', 'pdf_sha256': sha, 'txt': f'docs/papers/text/{dst.name}',
                'pages': len(pages), 'chars': len(body),
                'empty_pages': sum(1 for t in pages if len(t.strip()) < 20)})
    print(dst.name, len(pages), len(body), man[-1]['empty_pages'])
(out / 'MANIFEST.json').write_text(json.dumps({'extractor': f'pypdf {pypdf.__version__}', 'files': man}, indent=1))
