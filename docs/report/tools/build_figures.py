"""Generate the report architecture figures and editable TeX math assets."""
from pathlib import Path
from xml.etree import ElementTree as ET
import copy,json,re,urllib.parse
from pypdf import PdfReader,PdfWriter,Transformation
import pypdfium2 as pdfium
from math_renderer import MathRenderer,ROOT,TEMP,SVG_NS

import scene_base as base
base.OUT=ROOT
MATH_SPEC=json.loads((ROOT/'math_spec.json').read_text(encoding='utf8'))

def drop(s,predicate):s.items[:]=[i for i in s.items if not predicate(i)]
def text(s,value):return [i for i in s.items if i['kind']=='text' and i['text']==value]
def change(s,value,**kwargs):
    found=text(s,value);assert found,value
    for item in found:item.update(kwargs)
def drop_text(s,value):drop(s,lambda i:i['kind']=='text' and i['text']==value)
def edge(s,points):
    result=[i for i in s.items if i['kind']=='line' and i['points']==points]
    assert len(result)==1,points
    return result[0]

def refine06(s):
    change(s,'A',x=335,y=143)
    change(s,'B',x=335,y=233)
    # Restore the single evaluator; remove duplicated metric lists from task panels.
    drop_text(s,'BLEU / METEOR / ROUGE-L / CIDEr-D')
    drop_text(s,'Name F1 / weight & nutrition errors / validity')
    old=[i for i in s.items if i['kind']=='line' and i['dash'] and i['points'][0]==(647,125)]
    assert len(old)==1;s.items.remove(old[0])
    s.arrow([(647,185),(647,197),(321,197),(321,328),(315,328),(315,392),(460,392),(460,398)],'orangeD',True)
    # The only crossing is an image input / parameter-transfer overpass.
    s.circle(321,245,4,'white','white',0);s.arrow([(314,245),(328,245)],'line',head=False)
    s.rect(26,408,245,181,'gray','blueD',r=3,group='container')
    s.text(148,437,'Unified evaluation',20,True,'blueD')
    s.text(148,468,'Caption metrics\nIngredient F1\nWeight / nutrition errors\nJSON / Schema validity\nJudge-human agreement',18,color='ink')
    s.arrow([(950,155),(988,155),(988,610),(282,610),(282,500),(271,500)])
    s.arrow([(950,245),(988,245)],head=False)
    s.arrow([(950,478),(988,478)],head=False)
    s.circle(988,245,2,'line','line',0);s.circle(988,478,2,'line','line',0)
    # SCST continues A; Judge results contribute to evaluation.
    s.arrow([(271,143),(315,143),(315,110),(600,110),(600,125)],'orangeD',True)
    s.arrow([(26,184),(10,184),(10,385),(148,385),(148,408)],'tealD')
    # Branch above the JSON join: only captions enter the optional experiments.
    s.arrow([(988,300),(998,300),(998,50),(148,50),(148,99)],'tealD')
    s.circle(988,300,2,'tealD','tealD',0)
    s.text(845,43,'captions',18,color='tealD')
    return s

def refine07(s):
    s.text(871,139,'initial state',18,color='blueD')
    s.text(655,172,'at every step',18,color='blueD')
    s.text(457,457,'49 spatial tokens',18,color='muted')
    change(s,'224 × 224',x=77)
    change(s,'Q',x=578,y=527)
    change(s,'K, V',x=703)
    footer='Shifted targets: teacher forcing  |  Dashed feedback: autoregressive inference'
    change(s,footer,y=686,text='Teacher forcing; dashed feedback: autoregressive inference')
    s.text(500,713,'Sublayer schematic: residual connections and layer norms are omitted.',18,color='muted')
    return s

def refine08(s):
    s.h=824
    # Dashed arrows select time indices; they do not feed SEP embeddings to the head.
    for e in s.items:
        if e['kind']=='line' and e['points'] in [[(211,442),(211,474)],[(430,442),(430,474)]]:
            e['dash']=True;e['tag']='select_hidden_state_that_predicts_SEP'
    change(s,'State predicting SEP',text='GRU state predicting SEP')
    drop_text(s,'g / kcal / g / g / g')
    change(s,'q: 5 totals',y=487)
    s.text(750,517,'W  total_weight_g\nE  total_calories_kcal\nP  protein_g\nF  fat_g\nC  carbohydrate_g',18,anchor='start',color='muted',width=233)
    change(s,'[h(bk); z]: 512 -> shared MLP',y=674)
    for i in s.items:
        if i['kind']=='rect' and i['x']==26 and i['y']==750:i.update(y=742,h=45)
    raw='L = λ_ing L_ing + λ_w L_w + λ_q L_q   |   CE + standardized Smooth L1'
    change(s,raw,y=772,size=26)
    s.text(500,811,'CE: ingredient tokens; standardized Smooth L1: item weights and five meal totals.',18,color='muted',width=950)
    return s

def refine09(s):
    # Existing decomposition remains unchanged; add the exact low-rank equation.
    change(s,'Only low-rank parameters are updated',text='LoRA update equation',y=493,size=26)
    s.text(313,525,'Only low-rank matrices are trainable; base weights stay frozen.',18,color='orangeD',width=560)
    # Rebuild ONLY the lower comparison detail, retaining its two-row arrangement.
    drop(s,lambda i: i['kind']=='line' and min(y for x,y in i['points'])>=580)
    drop(s,lambda i: i['kind'] in {'rect','text'} and i.get('y',0)>=580)
    s.h=881
    s.panel(580,255,'(b) Matched inference and evaluation','tealD')
    s.block(28,665,190,65,'Image + fixed\ninstruction','gray',21)
    s.arrow([(218,697),(244,697)],head=False)
    s.arrow([(244,697),(244,657),(280,657)])
    s.arrow([(244,697),(244,747),(280,747)])
    s.block(280,635,220,44,'Original Instruct model','gray',19)
    s.block(280,725,220,44,'Same base + adapter','orange',20)
    s.text(642,612,'JSON schema sketch',19,True,'muted')
    for yy in [657,747]:
        s.arrow([(500,yy),(537,yy)])
        s.block(537,yy-35,211,70,'items: [{name: str,\nweight_g: num}]\n+ 5 numeric totals','gray',18)
    s.block(795,665,175,80,'JSON + Schema\nF1 / MAE / MSE','teal',18,stroke='tealD')
    s.arrow([(748,657),(770,657),(770,686),(795,686)])
    s.arrow([(748,747),(770,747),(770,724),(795,724)])
    s.text(500,863,'Same base revision, images, prompts, generation limits and metric protocol',18,color='muted',width=950)
    return s

class ArchitectureScene(base.Scene):
    @classmethod
    def from_base(cls,s,renderer):
        obj=cls(s.name,s.h,s.title);obj.items=copy.deepcopy(s.items);obj.renderer=renderer
        for n,item in enumerate(obj.items,1):item['id']=s.name+f'-v2-{n:03d}'
        obj.convert_math();return obj
    def convert_math(self):
        converted=[];audit=[]
        for i in self.items:
            if i['kind']!='text':converted.append(i);continue
            value=i['text']
            # A/B in Figure 6 identify models, not low-rank matrices.
            if value in ['A','B'] and self.name!='figure09_qwen':converted.append(i);continue
            tex=MATH_SPEC.get(value)
            if value in ['+','||','...']:
                tex={'+':'+','||':r'\Vert','...':r'\cdots'}[value]
            if value=='concat: 512':tex=r'\mathrm{concat}:512'
            if not tex:converted.append(i);continue
            j={k:v for k,v in i.items() if k!='text'}
            j.update(kind='math',tex=tex,original_text=value,color=base.C['ink'])
            if value in ['h(b1)','h(b2)','h(0) = z']:j['size']=26
            elif value in ['W0','A','B','α/r']:j['size']=24
            elif value in ['v: 512','z: 256','M: 49 × 256','q: 5 totals','768']:j['size']=23
            elif value.startswith('F: '):j['size']=21
            else:j['size']=max(j['size'],22)
            if value=='M: 49 × 256':j['x']=850
            if value=='768':j.update(x=920,y=310,size=22)
            if value=='49 spatial tokens':j.update(size=18,x=457)
            converted.append(j);audit.append({'id':i['id'],'before':value,'latex':tex,'math_asset':renderer_key(tex)})
        self.items=converted;self.audit=audit
    def save(self):
        (ROOT/(self.name+'.scene.json')).write_text(json.dumps({'name':self.name,'width':self.w,'height':self.h,'items':self.items},ensure_ascii=False,indent=2),encoding='utf8')
        all_items=self.items;self.items=[i for i in all_items if i['kind']!='math']
        base.Scene.save_svg(self);base.Scene.save_drawio(self)
        base.OUT=TEMP;base.Scene.save_pdf(self);base.OUT=ROOT
        self.items=all_items
        self.add_math_svg();self.add_math_drawio();self.add_math_pdf()
        doc=pdfium.PdfDocument(str(ROOT/(self.name+'.pdf')))
        doc[0].render(scale=300/72).to_pil().save(ROOT/(self.name+'.png'),dpi=(300,300))
    def add_math_svg(self):
        p=ROOT/(self.name+'.svg');tree=ET.parse(p);root=tree.getroot()
        for item in self.items:
            if item['kind']=='math':root.append(self.renderer.svg_group(item))
        ET.indent(root);tree.write(p,encoding='utf-8',xml_declaration=True)
    def add_math_pdf(self):
        writer=PdfWriter();writer.add_page(PdfReader(TEMP/(self.name+'.pdf')).pages[0]);p=writer.pages[0]
        scene_scale=16.3/2.54*72/1000
        for i in self.items:
            if i['kind']!='math':continue
            a=self.renderer.assets[i['tex']];x,y,w,h=self.renderer.bounds(i);s=i['size']/10*scene_scale
            tx=x*scene_scale;ty=(self.h-i['y'])*scene_scale-a['baseline']*s
            src=copy.deepcopy(PdfReader(ROOT/a['pdf']).pages[0])
            p.merge_transformed_page(src,Transformation().scale(s).translate(tx,ty))
        with (ROOT/(self.name+'.pdf')).open('wb') as f:writer.write(f)
    def add_math_drawio(self):
        p=ROOT/(self.name+'.drawio');tree=ET.parse(p);root=tree.find('.//root')
        for i in self.items:
            if i['kind']!='math':continue
            a=self.renderer.assets[i['tex']];svg=(ROOT/a['svg']).read_text(encoding='utf8')
            encoded=urllib.parse.quote(svg,safe='')
            x,y,w,h=self.renderer.bounds(i)
            style='shape=image;imageAspect=0;aspect=fixed;image=data:image/svg+xml,'+encoded+';'
            cell=ET.SubElement(root,'mxCell',{'id':i['id'],'parent':'1','vertex':'1','value':'','style':style,
                'latex':i['tex'],'formulaSource':'math/sources/'+a['id']+'.tex'})
            ET.SubElement(cell,'mxGeometry',{'x':str(x),'y':str(y),'width':str(w),'height':str(h),'as':'geometry'})
        ET.indent(tree);tree.write(p,encoding='utf8',xml_declaration=True)

def renderer_key(tex):
    from math_renderer import key
    return key(tex)

def main():
    extras={f'extra{i}':t for i,t in enumerate([r'\Vert','+',r'\cdots',r'\mathrm{concat}:512',r'h_{b_k}'])}
    renderer=MathRenderer(MATH_SPEC|extras).build()
    audit={}
    for key,refine in [('06',refine06),('07',refine07),('08',refine08),('09',refine09)]:
        scene=refine(getattr(base,'figure'+key)());s=ArchitectureScene.from_base(scene,renderer);s.save();audit[s.name]=s.audit
    (TEMP/'math_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
    print('Rendered four refined scenes using',len(renderer.assets),'shared TeX math assets.')

if __name__=='__main__':main()
