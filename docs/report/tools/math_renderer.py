"""One TeX source drives embedded PDF math and font-independent SVG paths."""
from pathlib import Path
from xml.etree import ElementTree as ET
import hashlib,json,os,re,shutil,subprocess
from pypdf import PdfReader,PdfWriter

REPORT_ROOT=Path(__file__).resolve().parents[1]
ROOT=REPORT_ROOT/'figures/architecture'
MATH=ROOT/'math'
TEMP=REPORT_ROOT/'build/figures'
WORK=TEMP/'math'
PT_TO_BP=72/72.27
SVG_NS='http://www.w3.org/2000/svg'

def key(tex):return 'f_'+hashlib.sha256(tex.encode()).hexdigest()[:12]

class MathRenderer:
    def __init__(self,spec):
        self.spec=spec;self.assets={}
    def build(self):
        (WORK/'sources').mkdir(parents=True, exist_ok=True)
        (MATH/'sources').mkdir(parents=True, exist_ok=True)
        expressions=sorted(set(self.spec.values()))
        header=r'''\documentclass{article}
\usepackage{amsmath,amssymb,xcolor}
\usepackage[active,tightpage]{preview}
\setlength\PreviewBorder{0.6pt}
\definecolor{FLMathInk}{HTML}{253747}
\ifdefined\pdfgentounicode\input{glyphtounicode}\pdfgentounicode=1\fi
\newbox\mathbox
\newwrite\metricfile
\immediate\openout\metricfile=atlas.metrics
\begin{document}
'''
        body=[]
        for n,tex in enumerate(expressions,1):
            ident=key(tex)
            source=r'{\color{FLMathInk}$\displaystyle '+tex+'$}'
            (MATH/'sources'/(ident+'.tex')).write_text(source+'\n',encoding='utf8')
            (WORK/'sources'/(ident+'.tex')).write_text(source+'\n',encoding='utf8')
            body += [rf'\setbox\mathbox=\hbox{{\input{{sources/{ident}.tex}}}}',
              rf'\immediate\write\metricfile{{{ident}|\the\wd\mathbox|\the\ht\mathbox|\the\dp\mathbox}}',
              r'\begin{preview}\copy\mathbox\end{preview}']
        document=header+'\n'.join(body)+'\n\\end{document}\n'
        (MATH/'atlas.tex').write_text(document,encoding='utf8')
        (WORK/'atlas.tex').write_text(document,encoding='utf8')
        commands=[['latex','-interaction=nonstopmode','-halt-on-error','atlas.tex'],
                  ['dvisvgm','--no-fonts','--bbox=preview','--page=1-','--output=page-%p.svg','atlas.dvi'],
                  ['pdflatex','-interaction=nonstopmode','-halt-on-error','atlas.tex']]
        for cmd in commands:
            if os.environ.get('TEX_BIN'):
                executable=Path(os.environ['TEX_BIN'])/(cmd[0]+('.exe' if os.name=='nt' else ''))
                if not executable.is_file():raise FileNotFoundError(executable)
            else:
                executable=shutil.which(cmd[0])
                if not executable:raise FileNotFoundError(f'{cmd[0]}: add TeX Live to PATH or set TEX_BIN')
            result=subprocess.run([str(executable),*cmd[1:]],cwd=WORK,capture_output=True)
            (TEMP/('math-'+cmd[0]+'.txt')).write_bytes(result.stdout+result.stderr)
            if result.returncode:raise RuntimeError(f'{cmd[0]} failed: see {TEMP}/math-{cmd[0]}.txt')
        metrics={}
        for line in (WORK/'atlas.metrics').read_text().splitlines():
            ident,w,h,d=line.split('|');metrics[ident]=[float(v.removesuffix('pt'))*PT_TO_BP for v in [w,h,d]]
        pdf=PdfReader(WORK/'atlas.pdf')
        for n,tex in enumerate(expressions,1):
            ident=key(tex);p=pdf.pages[n-1];width=float(p.mediabox.width);height=float(p.mediabox.height)
            page_svg=WORK/f'page-{n:0{len(str(len(expressions)))}}.svg'
            root=ET.parse(page_svg).getroot();view=list(map(float,root.attrib['viewBox'].split()))
            assert abs(view[2]-width)<.003 and abs(view[3]-height)<.003,(ident,view,width,height)
            target=MATH/(ident+'.svg');target.write_bytes(page_svg.read_bytes())
            writer=PdfWriter();writer.add_page(p)
            with (MATH/(ident+'.pdf')).open('wb') as f:writer.write(f)
            self.assets[tex]={'id':ident,'tex':tex,'width':width,'height':height,
                'baseline':metrics[ident][2]+.6*PT_TO_BP,'viewbox':view,'page':n,
                'svg':str(target.relative_to(ROOT)).replace('\\','/'),
                'pdf':str((MATH/(ident+'.pdf')).relative_to(ROOT)).replace('\\','/')}
        (MATH/'manifest.json').write_text(json.dumps(self.assets,ensure_ascii=False,indent=2),encoding='utf8')
        return self
    def bounds(self,item):
        a=self.assets[item['tex']];s=item['size']/10;w=a['width']*s;h=a['height']*s
        x=item['x']-{'start':0,'middle':w/2,'end':w}[item['anchor']]
        top=item['y']-(a['height']-a['baseline'])*s
        return x,top,w,h
    def svg_group(self,item):
        import copy
        a=self.assets[item['tex']];x,y,_,_=self.bounds(item);s=item['size']/10
        root=ET.parse(ROOT/a['svg']).getroot();g=ET.Element(f'{{{SVG_NS}}}g',
            {'id':item['id'],'data-latex':item['tex'],'transform':f'translate({x} {y}) scale({s})'})
        inner=ET.SubElement(g,f'{{{SVG_NS}}}g',{'transform':f'translate({-a["viewbox"][0]} {-a["viewbox"][1]})'})
        prefix=item['id']+'-'
        ids={e.get('id'):prefix+e.get('id') for e in root.iter() if e.get('id')}
        for el in root.iter():
            if el.get('id'):el.set('id',ids[el.get('id')])
            for attr,value in list(el.attrib.items()):
                if value.startswith('#') and value[1:] in ids:el.set(attr,'#'+ids[value[1:]])
                for before,after in ids.items():value=value.replace('url(#'+before+')','url(#'+after+')')
                if 'url(' in value:el.set(attr,value)
        for child in root:inner.append(copy.deepcopy(child))
        return g
