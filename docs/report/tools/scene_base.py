"""FoodLens editable vector scene -> native SVG, draw.io objects, PDF and PNG.

Coordinates use a top-left origin. Fonts remain text in SVG/draw.io/PDF.
No external artwork, templates, model inference, or downloaded fonts are used.
"""
from pathlib import Path
import argparse, base64, hashlib, html, json, math, zlib
from xml.etree import ElementTree as ET
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
import pypdfium2 as pdfium

OUT=Path(__file__).resolve().parents[1]/'figures/architecture'
ROOT=OUT.parents[1]
C={'ink':'#253747','muted':'#607382','line':'#657C8C','border':'#B3C2CD',
   'blue':'#DCEAF4','blueD':'#3C6B8C','teal':'#DDEFEA','tealD':'#3B7D71',
   'violet':'#E9E3F1','violetD':'#76618F','orange':'#F7E5CE','orangeD':'#AA7439',
   'gray':'#F2F5F7','pale':'#FAFBFC','white':'#FFFFFF'}
pdfmetrics.registerFont(TTFont('Arial',r'C:\Windows\Fonts\arial.ttf'))
pdfmetrics.registerFont(TTFont('Arial-Bold',r'C:\Windows\Fonts\arialbd.ttf'))

class Scene:
    def __init__(self,name,height,title):
        self.name,self.w,self.h,self.title=name,1000,height,title
        self.items=[];self.edges=[];self.text_boxes=[]
    def add(self,kind,**kw):
        kw.update(kind=kind,id=f'{self.name}-{len(self.items)+1:03d}')
        self.items.append(kw);return kw
    def rect(self,x,y,w,h,fill='white',stroke='border',r=3,sw=1.4,dash=False,group=''):
        return self.add('rect',x=x,y=y,w=w,h=h,fill=C.get(fill,fill),stroke=C.get(stroke,stroke),r=r,sw=sw,dash=dash,group=group)
    def poly(self,pts,fill='white',stroke='border',sw=1.2,group=''):
        return self.add('poly',points=pts,fill=C.get(fill,fill),stroke=C.get(stroke,stroke),sw=sw,group=group)
    def circle(self,x,y,r,fill='white',stroke='line',sw=1.4):
        return self.add('circle',x=x,y=y,r=r,fill=C.get(fill,fill),stroke=C.get(stroke,stroke),sw=sw)
    def text(self,x,y,value,size=21,bold=False,color='ink',anchor='middle',width=None,group=''):
        value=value.replace(' x ', ' × ')
        lines=value.split('\n'); font='Arial-Bold' if bold else 'Arial'
        for j,line in enumerate(lines):
            actual=pdfmetrics.stringWidth(line,font,size)
            if width and actual>width+0.1: raise ValueError(f'Text too wide: {line}: {actual:.1f}>{width}')
            self.add('text',x=x,y=y+j*size*1.18,text=line,size=size,bold=bold,color=C.get(color,color),anchor=anchor,group=group)
        return y+len(lines)*size*1.18
    def arrow(self,pts,color='line',dash=False,sw=1.8,head=True,tag=''):
        obj=self.add('line',points=pts,color=C.get(color,color),dash=dash,sw=sw,head=head,tag=tag)
        self.edges.append(obj);return obj
    def block(self,x,y,w,h,label,fill='blue',size=21,stroke=None,stack=0,bold=False):
        gid=f'component-{len(self.items)}'
        for k in range(stack,0,-1): self.rect(x+k*5,y-k*5,w,h,fill,stroke or 'border',group=gid)
        self.rect(x,y,w,h,fill,stroke or 'border',group=gid)
        n=label.count('\n')+1
        self.text(x+w/2,y+h/2-(n-1)*size*.59+size*.34,label,size,bold,width=w-10,group=gid)
    def tokens(self,x,y,labels,fill='teal',cw=46,ch=34,gap=5,size=18):
        gid=f'component-{len(self.items)}'
        for i,label in enumerate(labels):
            self.rect(x+i*(cw+gap),y,cw,ch,fill,'white',r=1,sw=1,group=gid)
            self.text(x+i*(cw+gap)+cw/2,y+ch/2+size*.34,label,size,width=cw-3,group=gid)
        return len(labels)*(cw+gap)-gap
    def bars(self,x,y,n=5,w=12,h=53,fill='blue'):
        gid=f'component-{len(self.items)}'
        for k in range(n): self.rect(x+k*(w+3),y,w,h,fill,'white',r=0,sw=.8,group=gid)
    def tensor(self,x,y,size=64,depth=15,grid=7):
        start=len(self.items)
        for k in [2,1,0]:
            self.rect(x+k*depth/2,y-k*depth/2,size,size,'blue','blueD',r=0,sw=1)
        for k in range(1,grid):
            a=k*size/grid
            self.arrow([(x+a,y),(x+a,y+size)],color='border',sw=.65,head=False)
            self.arrow([(x,y+a),(x+size,y+a)],color='border',sw=.65,head=False)
        for item in self.items[start:]:item['group']=f'component-{start}'
    def backbone(self,x,y,w=115,h=64):
        self.block(x,y,w,h,'ResNet-18','blue',size=19 if w<110 else 21,stroke='blueD',stack=2,bold=True)
    def image(self,x,y,size=62):
        start=len(self.items)
        self.rect(x,y,size,size,'gray','border',r=1)
        self.circle(x+size/2,y+size/2,size*.34,'white','border',1)
        for dx,dy,rr,col in [(-10,-3,9,'orange'),(8,6,8,'teal'),(7,-11,6,'teal'),(-8,11,5,'orange')]:
            self.circle(x+size/2+dx,y+size/2+dy,rr,col,col,1)
        for item in self.items[start:]:item['group']=f'component-{start}'
    def panel(self,y,h,label,color='blueD'):
        self.rect(8,y,984,h,'white','border',r=0,sw=1)
        self.rect(8,y,5,h,color,color,r=0,sw=0)
        self.text(26,y+28,label,24,True,color,anchor='start',width=940)
    def legend(self,y):
        self.rect(26,y-11,16,16,'blue','blueD',r=1)
        self.text(50,y+3,'visual',18,color='muted',anchor='start')
        self.rect(145,y-11,16,16,'teal','tealD',r=1)
        self.text(169,y+3,'sequence / attention',18,color='muted',anchor='start')
        self.rect(380,y-11,16,16,'violet','violetD',r=1)
        self.text(404,y+3,'regression',18,color='muted',anchor='start')
        self.arrow([(580,y-2),(620,y-2)],'orangeD',True)
        self.text(630,y+3,'transfer / supervision',18,color='muted',anchor='start')
    def save(self):
        scene={'name':self.name,'width':self.w,'height':self.h,'title':self.title,'items':self.items}
        (OUT/(self.name+'.scene.json')).write_text(json.dumps(scene,ensure_ascii=False,indent=2),encoding='utf8')
        self.save_svg();self.save_pdf();self.save_drawio()
        doc=pdfium.PdfDocument(str(OUT/(self.name+'.pdf')))
        doc[0].render(scale=300/72).to_pil().save(OUT/(self.name+'.png'),dpi=(300,300))
    def save_svg(self):
        ns='http://www.w3.org/2000/svg';ET.register_namespace('',ns)
        root=ET.Element('svg',{'xmlns':ns,'width':str(self.w),'height':str(self.h),'viewBox':f'0 0 {self.w} {self.h}',
            'role':'img','aria-label':self.title})
        ET.SubElement(root,'title').text=self.title
        ET.SubElement(root,'rect',{'width':str(self.w),'height':str(self.h),'fill':'white'})
        groups={}
        for i in self.items:
            parent=root
            gid=i.get('group','')
            if gid.startswith('component-'):
                if gid not in groups:groups[gid]=ET.SubElement(root,'g',{'id':gid,'data-editable':'true'})
                parent=groups[gid]
            common={'id':i['id']}
            k=i['kind']
            if k=='rect':
                a=common|{p:str(i[p]) for p in ['x','y']}
                a.update(width=str(i['w']),height=str(i['h']),rx=str(i['r']),fill=i['fill'],stroke=i['stroke'],**{'stroke-width':str(i['sw'])})
                if i['dash']:a['stroke-dasharray']='6 4'
                ET.SubElement(parent,'rect',a)
            elif k=='circle': ET.SubElement(parent,'circle',common|{'cx':str(i['x']),'cy':str(i['y']),'r':str(i['r']),'fill':i['fill'],'stroke':i['stroke'],'stroke-width':str(i['sw'])})
            elif k=='poly': ET.SubElement(parent,'polygon',common|{'points':' '.join(f'{x},{y}' for x,y in i['points']),'fill':i['fill'],'stroke':i['stroke'],'stroke-width':str(i['sw'])})
            elif k=='text':
                t=ET.SubElement(parent,'text',common|{'x':str(i['x']),'y':str(i['y']),'font-family':'Arial, sans-serif','font-size':str(i['size']),
                    'font-weight':'700' if i['bold'] else '400','fill':i['color'],'text-anchor':i['anchor']})
                t.text=i['text']
            elif k=='line':
                a=common|{'points':' '.join(f'{x},{y}' for x,y in i['points']),'fill':'none','stroke':i['color'],'stroke-width':str(i['sw']),'stroke-linejoin':'round'}
                if i['dash']:a['stroke-dasharray']='6 4'
                ET.SubElement(parent,'polyline',a)
                if i['head']:
                    p=arrowhead(i['points']); ET.SubElement(parent,'polygon',{'points':' '.join(f'{x},{y}' for x,y in p),'fill':i['color']})
        ET.indent(root);ET.ElementTree(root).write(OUT/(self.name+'.svg'),encoding='utf-8',xml_declaration=True)
    def save_pdf(self):
        # 1000 scene units occupy exactly 16.3 cm, matching the report text width.
        scale=(16.3/2.54*72)/self.w
        c=canvas.Canvas(str(OUT/(self.name+'.pdf')),pagesize=(self.w*scale,self.h*scale),pageCompression=1)
        c.setTitle(self.title);c.setAuthor('FoodLens')
        c.scale(scale,scale)
        for i in self.items:
            k=i['kind'];c.setDash(6,4) if i.get('dash') else c.setDash()
            if k in ['rect','circle','poly']:
                c.setFillColor(HexColor(i['fill']));c.setStrokeColor(HexColor(i['stroke']));c.setLineWidth(i['sw'])
                if k=='rect':c.roundRect(i['x'],self.h-i['y']-i['h'],i['w'],i['h'],i['r'],stroke=bool(i['sw']),fill=1)
                elif k=='circle':c.circle(i['x'],self.h-i['y'],i['r'],stroke=1,fill=1)
                else:
                    p=c.beginPath();p.moveTo(i['points'][0][0],self.h-i['points'][0][1])
                    for x,y in i['points'][1:]:p.lineTo(x,self.h-y)
                    p.close();c.drawPath(p,stroke=1,fill=1)
            elif k=='text':
                c.setFillColor(HexColor(i['color']));c.setFont('Arial-Bold' if i['bold'] else 'Arial',i['size'])
                f={'start':c.drawString,'middle':c.drawCentredString,'end':c.drawRightString}[i['anchor']]
                f(i['x'],self.h-i['y'],i['text'])
            else:
                c.setStrokeColor(HexColor(i['color']));c.setFillColor(HexColor(i['color']));c.setLineWidth(i['sw'])
                p=c.beginPath();p.moveTo(i['points'][0][0],self.h-i['points'][0][1])
                for x,y in i['points'][1:]:p.lineTo(x,self.h-y)
                c.drawPath(p,stroke=1)
                if i['head']:
                    c.setDash();pts=arrowhead(i['points']);p=c.beginPath();p.moveTo(pts[0][0],self.h-pts[0][1])
                    for x,y in pts[1:]:p.lineTo(x,self.h-y)
                    p.close();c.drawPath(p,stroke=0,fill=1)
        c.showPage();c.save()
    def save_drawio(self):
        mx=ET.Element('mxfile',{'host':'app.diagrams.net','version':'24.7.17'})
        dia=ET.SubElement(mx,'diagram',{'id':self.name,'name':self.title})
        model=ET.SubElement(dia,'mxGraphModel',{'dx':'1000','dy':str(self.h),'grid':'1','gridSize':'5','page':'1','pageWidth':'1000','pageHeight':str(self.h),'background':'#ffffff'})
        root=ET.SubElement(model,'root');ET.SubElement(root,'mxCell',{'id':'0'});ET.SubElement(root,'mxCell',{'id':'1','parent':'0'})
        def box(i):
            if i['kind']=='rect':return i['x'],i['y'],i['w'],i['h']
            if i['kind']=='circle':return i['x']-i['r'],i['y']-i['r'],i['r']*2,i['r']*2
            if i['kind']=='text':
                w=pdfmetrics.stringWidth(i['text'],'Arial-Bold' if i['bold'] else 'Arial',i['size'])+8
                return i['x']-{'start':0,'middle':w/2,'end':w}[i['anchor']],i['y']-i['size']*.95,w,i['size']*1.24
            xs,ys=zip(*i['points']);return min(xs),min(ys),max(xs)-min(xs),max(ys)-min(ys)
        group_boxes={}
        for i in self.items:
            g=i.get('group','')
            if not g.startswith('component-'):continue
            x,y,w,h=box(i)
            if g not in group_boxes:group_boxes[g]=[x,y,x+w,y+h]
            else:
                a=group_boxes[g];a[:]=[min(a[0],x),min(a[1],y),max(a[2],x+w),max(a[3],y+h)]
        made=set()
        for i in self.items:
            gid=i.get('group','');offset=(0,0);parent='1'
            if gid in group_boxes:
                parent=gid;a,b,c,d=group_boxes[gid];offset=(a,b)
                if gid not in made:
                    gr=ET.SubElement(root,'mxCell',{'id':gid,'parent':'1','value':'','vertex':'1','style':'group;','connectable':'1'})
                    ET.SubElement(gr,'mxGeometry',{'x':str(a),'y':str(b),'width':str(c-a),'height':str(d-b),'as':'geometry'});made.add(gid)
            k=i['kind'];base={'id':i['id'],'parent':parent}
            if k=='line':
                cell=ET.SubElement(root,'mxCell',base|{'edge':'1','value':'','style':f"edgeStyle=none;rounded=0;endArrow={'block' if i['head'] else 'none'};endFill=1;strokeColor={i['color']};strokeWidth={i['sw']};dashed={int(i['dash'])};"})
                g=ET.SubElement(cell,'mxGeometry',{'relative':'1','as':'geometry'})
                for role,(x,y) in [('sourcePoint',i['points'][0]),('targetPoint',i['points'][-1])]:ET.SubElement(g,'mxPoint',{'x':str(x-offset[0]),'y':str(y-offset[1]),'as':role})
                if len(i['points'])>2:
                    arr=ET.SubElement(g,'Array',{'as':'points'})
                    for x,y in i['points'][1:-1]:ET.SubElement(arr,'mxPoint',{'x':str(x-offset[0]),'y':str(y-offset[1])})
                continue
            value='';style=''
            if k=='text':
                font='Arial-Bold' if i['bold'] else 'Arial';w=pdfmetrics.stringWidth(i['text'],font,i['size'])+8;h=i['size']*1.24
                x=i['x']-({'start':0,'middle':w/2,'end':w}[i['anchor']]);y=i['y']-i['size']*.95
                value=i['text'];style=f"text;html=0;whiteSpace=wrap;overflow=visible;align={dict(start='left',middle='center',end='right')[i['anchor']]};verticalAlign=middle;fontFamily=Arial;fontSize={i['size']};fontColor={i['color']};fontStyle={int(i['bold'])};spacing=0;"
            elif k=='rect':
                x,y,w,h=i['x'],i['y'],i['w'],i['h'];style=f"rounded={int(i['r']>0)};arcSize=5;"
            elif k=='circle':
                x,y,w,h=i['x']-i['r'],i['y']-i['r'],i['r']*2,i['r']*2;style='ellipse;'
            else:
                # All current scientific primitives are rectangles, circles or lines.
                # SVG retains arbitrary polygon geometry for future extension.
                xs,ys=zip(*i['points']);x,y=min(xs),min(ys);w,h=max(xs)-x,max(ys)-y
                style='shape=rectangle;'
            if k!='text':style+=f"fillColor={i['fill']};strokeColor={i['stroke']};strokeWidth={i['sw']};dashed={int(i.get('dash',False))};"
            cell=ET.SubElement(root,'mxCell',base|{'vertex':'1','value':value,'style':style})
            ET.SubElement(cell,'mxGeometry',{'x':str(x-offset[0]),'y':str(y-offset[1]),'width':str(w),'height':str(h),'as':'geometry'})
        ET.indent(mx);ET.ElementTree(mx).write(OUT/(self.name+'.drawio'),encoding='utf-8',xml_declaration=True)

def arrowhead(points):
    (x0,y0),(x,y)=points[-2:];angle=math.atan2(y-y0,x-x0);size=8;half=3.3
    return [(x,y),(x-size*math.cos(angle)+half*math.sin(angle),y-size*math.sin(angle)-half*math.cos(angle)),
            (x-size*math.cos(angle)-half*math.sin(angle),y-size*math.sin(angle)+half*math.cos(angle))]

def figure06():
    s=Scene('figure06_system',657,'FoodLens: task relationships and shared infrastructure')
    s.text(26,31,'FoodLens',27,True,'blueD',anchor='start')
    s.rect(330,55,650,245,'pale','border',r=3,group='container')
    s.text(348,86,'TASK 1  |  Image captioning',23,True,'blueD',anchor='start')
    s.rect(330,345,650,250,'pale','border',r=3,group='container')
    s.text(348,376,'TASK 2  |  Structured meal records',23,True,'violetD',anchor='start')
    s.image(28,263);s.text(59,355,'RGB',21)
    s.block(125,257,130,75,'Data version\nPreprocess','gray',18)
    s.arrow([(90,294),(125,294)]);s.arrow([(255,294),(300,294)],head=False)
    s.arrow([(300,155),(300,540)],head=False)
    for yy in [155,245,433,537]:s.arrow([(300,yy),(355,yy)])
    for yy in [125,215]:s.backbone(355,yy,105,60)
    s.text(367,121,'A',18,True,'blueD',anchor='start')
    s.text(367,211,'B',18,True,'blueD',anchor='start')
    s.bars(499,133,n=4,w=10,h=43)
    s.tensor(500,225,43,8,4)
    s.block(585,125,125,60,'GRU','teal',23,bold=True)
    s.block(585,215,125,60,'Transformer','teal',19,bold=True)
    for yy in [155,245]:
        s.arrow([(460,yy),(499,yy)]);s.arrow([(551,yy),(585,yy)]);s.arrow([(710,yy),(765,yy)])
    s.block(765,130,185,50,'Caption A','gray',21)
    s.block(765,220,185,50,'Caption B','gray',21)
    s.text(655,291,'BLEU / METEOR / ROUGE-L / CIDEr-D',18,color='muted')
    # Parameter transport follows the outer routing channel, not a data edge.
    s.arrow([(647,125),(647,105),(970,105),(970,328),(315,328),(315,392),(460,392),(460,398)],'orangeD',True)
    s.text(665,319,'A: ResNet + projection + GRU',18,color='orangeD')
    s.block(355,398,215,70,'Transferred A\nIngredient sequence','teal',20,bold=True)
    s.block(613,398,105,70,'Weight\n+ totals','violet',20)
    s.arrow([(570,433),(613,433)],'violetD')
    s.block(355,510,340,55,'Qwen3-VL + LoRA / QLoRA','orange',20,bold=True)
    s.block(775,435,175,87,'JSON\nitems + 5 totals\nSchema check','gray',18)
    s.arrow([(718,433),(745,433),(745,461),(775,461)])
    s.arrow([(695,537),(745,537),(745,494),(775,494)])
    s.text(655,583,'Name F1 / weight & nutrition errors / validity',18,color='muted')
    s.rect(26,99,245,113,'white','border',dash=True)
    s.text(40,124,'Optional experiments',20,True,'muted',anchor='start')
    s.text(40,153,'SCST: continue training A\nJudge: score captions\nHuman agreement checks',18,color='muted',anchor='start',width=220)
    s.legend(635)
    return s

def figure07():
    s=Scene('figure07_caption',722,'Captioning models: global GRU and grid Transformer')
    s.panel(8,280,'(a) Global visual conditioning + GRU')
    s.image(28,75);s.text(59,165,'RGB',20);s.text(59,188,'224 x 224',18,color='muted')
    s.backbone(125,75);s.text(183,165,'layer4',19,color='muted')
    s.tensor(280,75);s.text(312,165,'F: 512 x 7 x 7',19)
    s.block(390,86,70,44,'GAP','blue',20)
    s.bars(496,85,n=4,w=10,h=45);s.text(522,165,'v: 512',19)
    s.block(585,78,125,60,'Linear\ntanh','blue',20)
    s.bars(750,85,n=4,w=10,h=45,fill='blue');s.text(775,165,'z: 256',19)
    for a,b in [((90,107),(125,107)),((240,107),(280,107)),((359,107),(390,107)),((460,107),(496,107)),((548,107),(585,107)),((710,107),(750,107))]:s.arrow([a,b])
    s.tokens(28,218,['BOS','...','y(t-1)'],cw=66,ch=34,size=18)
    s.arrow([(236,235),(260,235)])
    s.block(260,210,132,51,'Embedding\n256','teal',19)
    s.arrow([(392,235),(446,235)])
    s.circle(462,235,16,'white','tealD');s.text(462,242,'||',22)
    s.text(462,275,'concat: 512',18,color='muted')
    s.arrow([(478,235),(530,235)])
    s.block(530,207,135,55,'GRU 256','teal',22,stroke='tealD',bold=True)
    s.arrow([(665,235),(716,235)])
    s.block(716,210,106,51,'Linear\nSoftmax','teal',18)
    s.arrow([(822,235),(850,235)])
    s.tokens(850,218,['y(t)'],cw=70,ch=34,size=20)
    # z is supplied to the concatenation at EVERY autoregressive step.
    s.arrow([(775,130),(775,142),(725,142),(725,181),(462,181),(462,219)],'blueD')
    s.arrow([(800,107),(946,107),(946,191),(597,191),(597,207)],'blueD')
    s.text(877,94,'h(0) = z',19,color='blueD')
    s.arrow([(886,252),(886,282),(318,282),(318,261)],color='tealD',dash=True)
    s.text(930,259,'AR',18,color='tealD')
    s.panel(308,352,'(b) Grid features + Transformer','tealD')
    s.image(28,369);s.text(59,457,'RGB',20)
    s.backbone(125,369);s.text(183,457,'layer4',19,color='muted')
    s.tensor(280,369);s.text(312,457,'F: 512 x 7 x 7',19)
    s.block(390,365,123,65,'Flatten\nLinear 256','blue',19)
    s.circle(548,399,15,'white','blueD');s.text(548,406,'+',24)
    s.text(548,350,'2D pos.',18,color='muted')
    s.arrow([(548,356),(548,384)],'blueD')
    s.block(595,355,164,83,'Self-Attn\nFFN','teal',20,stroke='tealD',stack=1)
    s.text(677,341,'Encoder x 2',21,True)
    s.bars(811,370,n=6,w=14,h=53,fill='blue');s.text(865,457,'M: 49 x 256',19)
    for a,b in [((90,401),(125,401)),((240,401),(280,401)),((359,401),(390,401)),((513,399),(533,399)),((563,399),(595,399)),((759,399),(811,399))]:s.arrow([a,b])
    s.tokens(28,536,['BOS','y(1)','...'],cw=57,ch=34,size=18)
    s.arrow([(209,553),(237,553)])
    s.block(237,521,154,65,'Embedding 256\n+ 1D position','teal',19)
    s.rect(420,488,438,115,'pale','tealD',r=3)
    s.text(436,513,'Decoder x 2',21,True,anchor='start')
    s.block(435,533,130,51,'Masked\nSelf-Attn','teal',19)
    s.block(591,533,130,51,'Cross-Attn','teal',20,stroke='tealD')
    s.block(746,533,96,51,'FFN','teal',20)
    s.arrow([(391,553),(435,553)]);s.arrow([(565,553),(591,553)]);s.arrow([(721,553),(746,553)])
    s.text(577,535,'Q',18,color='tealD')
    s.arrow([(865,423),(935,423),(935,475),(656,475),(656,533)],'blueD')
    s.text(681,526,'K, V',18,color='blueD')
    s.arrow([(842,553),(891,553)])
    s.block(891,523,83,67,'Linear\nSoftmax','teal',19)
    s.arrow([(933,590),(933,606)])
    s.tokens(897,606,['y(t)'],cw=73,ch=32,size=20)
    s.text(850,626,'greedy',18,color='muted')
    s.text(34,608,'Causal + PAD mask',18,color='muted',anchor='start')
    s.arrow([(970,622),(985,622),(985,650),(314,650),(314,586)],'tealD',True)
    s.text(500,708,'Shifted targets: teacher forcing  |  Dashed feedback: autoregressive inference',18,color='muted',width=950)
    return s

def figure08():
    s=Scene('figure08_hybrid',790,'Transferred sequence decoder with aligned ingredient regression')
    s.rect(20,14,960,67,'pale','border',r=3,group='container')
    s.text(40,40,'INITIALIZE FROM TASK 1 A',21,True,'orangeD',anchor='start')
    s.text(40,66,'ResNet-18 + visual projection + GRU; copy compatible vocabulary rows',20,anchor='start',width=920)
    s.arrow([(190,81),(190,142)],'orangeD',True)
    s.arrow([(628,81),(628,142)],'orangeD',True)
    s.arrow([(20,50),(9,50),(9,275),(510,275),(510,289)],'orangeD',True)
    s.image(28,148);s.text(59,236,'RGB',20)
    s.backbone(132,148,118,64)
    s.tensor(289,148,58,12,7);s.text(318,235,'512 x 7 x 7',19)
    s.block(387,158,70,44,'GAP','blue',20)
    s.bars(490,158,n=4,w=10,h=44);s.text(515,235,'v: 512',19)
    s.block(574,145,126,66,'Linear\ntanh','blue',20)
    s.bars(746,158,n=4,w=10,h=44);s.text(772,236,'z: 256',19)
    for a,b in [((90,180),(132,180)),((250,180),(289,180)),((359,180),(387,180)),((457,180),(490,180)),((542,180),(574,180)),((700,180),(746,180))]:s.arrow([a,b])
    s.tokens(28,300,['S(t-1)'],cw=96,ch=40,size=20)
    s.arrow([(124,320),(165,320)])
    s.block(165,290,133,60,'Embedding\n256','teal',20)
    s.arrow([(298,320),(342,320)])
    s.circle(358,320,16,'white','tealD');s.text(358,327,'||',22)
    s.arrow([(374,320),(420,320)])
    s.block(420,289,180,65,'GRU 256\nh(0) = z','teal',22,bold=True,stroke='tealD')
    s.text(358,359,'concat: 512',18,color='muted')
    s.arrow([(772,202),(772,215),(715,215),(715,259),(358,259),(358,304)],'blueD')
    # The independent meal head reads v and z; no generated name enters it.
    s.arrow([(542,180),(555,180),(555,247),(847,247),(847,286),(864,286)],'blueD')
    s.arrow([(798,180),(880,180),(880,270)],'blueD')
    s.circle(715,247,4,'white','white',0)
    s.arrow([(715,240),(715,254)],'blueD',head=False)
    s.circle(880,286,16,'white','violetD');s.text(880,293,'||',22)
    s.arrow([(880,302),(880,319)],'violetD')
    s.block(792,319,176,65,'MLP 256\nSoftplus','violet',21,stroke='violetD')
    s.text(920,310,'768',18,color='muted')
    s.arrow([(880,384),(880,411)],'violetD')
    s.tokens(791,411,['W','E','P','F','C'],fill='violet',cw=32,ch=44,gap=4,size=19)
    s.text(880,483,'q: 5 totals',21,True,'violetD')
    s.text(880,512,'g / kcal / g / g / g',18,color='muted')
    s.rect(25,392,522,65,'pale','border',r=2,group='container')
    s.text(31,383,'Ingredient sequence (multi-word names)',20,True,'tealD',anchor='start')
    labels=['BOS','n1','SEP','n2a','n2b','SEP','EOS']
    for k,label in enumerate(labels):s.tokens(32+k*73,408,[label],fill='orange' if label=='SEP' else 'teal',cw=67,ch=34,size=19)
    s.arrow([(510,354),(510,392)],'tealD')
    s.text(550,380,'vocab projection',18,color='muted',anchor='start')
    for xx,label in [(211,'h(b1)'),(430,'h(b2)')]:
        s.arrow([(xx,442),(xx,474)],'orangeD')
        s.block(xx-42,474,84,40,label,'orange',19)
    s.arrow([(253,494),(280,494),(280,551),(530,551)],'violetD')
    s.arrow([(472,494),(496,494),(496,534),(530,534)],'violetD')
    s.block(530,518,156,66,'Shared MLP 256\nSoftplus','violet',19,stroke='violetD')
    s.arrow([(715,259),(715,498),(608,498),(608,518)],'blueD')
    s.text(609,613,'w1, w2, ... (g)',20,True,'violetD')
    s.arrow([(608,584),(608,592)],'violetD')
    s.text(277,640,'State predicting SEP\n[h(bk); z]: 512 -> shared MLP',19,color='orangeD')
    s.block(748,650,222,68,'JSON assembly\n+ Schema validation','gray',20,bold=True)
    s.arrow([(608,621),(608,684),(748,684)],'violetD')
    s.arrow([(880,455),(985,455),(985,684),(970,684)],'violetD')
    s.arrow([(25,424),(15,424),(15,735),(850,735),(850,718)],'tealD')
    s.text(300,726,'generated names',19,color='tealD')
    s.rect(26,750,944,31,'orange','orange',r=2)
    s.text(498,772,'L = λ_ing L_ing + λ_w L_w + λ_q L_q   |   CE + standardized Smooth L1',18,color='ink',width=925)
    return s

def figure09():
    s=Scene('figure09_qwen',811,'Qwen3-VL parameter-efficient adaptation and paired evaluation')
    s.panel(8,553,'(a) Domain adaptation  |  Qwen3-VL-4B-Instruct','orangeD')
    s.image(29,84);s.text(60,177,'RGB',20)
    s.block(135,77,168,70,'Vision encoder\nFROZEN','blue',21,stroke='blueD')
    s.block(349,77,131,70,'Visual mapper\nFROZEN','blue',18,stroke='blueD')
    s.bars(519,89,n=5,w=13,h=51);s.text(554,178,'visual tokens',18,color='blueD')
    for a,b in [((91,115),(135,115)),((303,115),(349,115)),((480,115),(519,115))]:s.arrow([a,b])
    s.tokens(29,219,['prompt','...'],cw=77,ch=38,size=19)
    s.arrow([(188,238),(226,238)])
    s.block(226,208,154,62,'Tokenizer\n+ chat template','teal',19)
    s.bars(421,218,n=5,w=13,h=41,fill='teal');s.text(461,198,'text tokens',18,color='tealD')
    s.arrow([(380,238),(421,238)])
    s.rect(725,77,245,189,'gray','line',r=4,group='container')
    s.text(848,112,'Qwen3 language model',20,True,width=228)
    s.text(848,144,'Frozen base weights',19,color='muted')
    s.block(748,169,198,64,'Attention Q / K / V / O\n+ trainable LoRA','orange',18,stroke='orangeD')
    s.arrow([(596,115),(640,115),(640,183),(725,183)],'blueD')
    s.arrow([(498,238),(725,238)],'tealD')
    s.text(638,210,'fused inputs',18,color='muted')
    # Low-rank update detail: frozen W0 path + BA path; r is an initial design.
    s.rect(650,305,320,152,'pale','orangeD',r=3,group='container')
    s.text(666,330,'Low-rank projection update',19,True,'orangeD',anchor='start')
    s.block(700,347,80,38,'W0','gray',20)
    s.text(740,404,'frozen',18,color='muted')
    s.block(700,416,48,30,'A','orange',19,stroke='orangeD')
    s.block(789,416,48,30,'B','orange',19,stroke='orangeD')
    s.circle(907,366,15,'white','orangeD');s.text(907,373,'+',24)
    s.arrow([(670,366),(700,366)])
    s.arrow([(780,366),(892,366)])
    s.arrow([(682,366),(682,431),(700,431)],'orangeD')
    s.arrow([(748,431),(789,431)],'orangeD')
    s.arrow([(837,431),(907,431),(907,381)],'orangeD')
    s.text(866,421,'α/r',18,color='orangeD')
    s.arrow([(922,366),(951,366)])
    s.arrow([(847,233),(847,285),(810,285),(810,305)],'orangeD',True)
    s.rect(28,325,566,119,'pale','border',r=3,group='container')
    s.text(45,353,'Supervision mask',21,True,anchor='start')
    s.block(45,369,220,39,'image / user / PAD','gray',18)
    s.block(283,369,292,39,'assistant JSON + EOS','orange',20)
    s.text(155,433,'labels = -100',18,color='muted')
    s.text(430,433,'supervised answer tokens',18,color='orangeD')
    s.arrow([(429,369),(429,295),(670,295),(670,238),(725,238)],'tealD')
    s.text(550,285,'shifted answer prefix',18,color='tealD')
    s.block(650,486,320,55,'Assistant-only cross-entropy','orange',20,stroke='orangeD',bold=True)
    s.arrow([(594,390),(620,390),(620,513),(650,513)],'orangeD')
    s.arrow([(970,200),(985,200),(985,513),(970,513)])
    s.arrow([(810,486),(810,446)],'orangeD',True)
    s.arrow([(810,475),(724,475),(724,446)],'orangeD',True)
    s.text(313,493,'Only low-rank parameters are updated',20,color='orangeD')
    s.text(26,550,'Optional QLoRA: NF4 frozen weights; higher-precision compute',18,color='muted',anchor='start',width=930)
    s.panel(580,187,'(b) Matched inference and evaluation','tealD')
    s.block(28,650,190,65,'Image + fixed\ninstruction','gray',21)
    s.arrow([(218,682),(244,682)],head=False)
    s.arrow([(244,682),(244,652),(285,652)])
    s.arrow([(244,682),(244,717),(285,717)])
    s.block(285,630,240,44,'Original Instruct model','gray',20)
    s.block(285,695,240,44,'Same base + adapter','orange',20)
    for yy in [652,717]:
        s.arrow([(525,yy),(565,yy)])
        s.block(565,yy-21,64,42,'{ }','gray',24)
    s.block(740,650,230,72,'Strict JSON + Schema\nF1 / MAE / MSE','teal',20,stroke='tealD')
    s.arrow([(629,652),(690,652),(690,670),(740,670)])
    s.arrow([(629,717),(690,717),(690,703),(740,703)])
    s.text(500,794,'Same base revision, images, prompts, generation limits and metric protocol',18,color='muted',width=950)
    return s

