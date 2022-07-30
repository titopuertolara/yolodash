import plotly.express as px
import plotly.graph_objects as go
#from jupyter_dash import JupyterDash
from dash import Dash, dcc, html, Input, Output
import os
import cv2
import numpy as np
import torch
import time
class yolo_detector:
  def __init__(self,model,url):
    self.model=model
    self.url=url
    self.cap= cv2.VideoCapture(self.url)
  def capture(self):
    try:
      self.ret, self.frame = self.cap.read()
    except Exception as e:
      print(e)
      self.cap= cv2.VideoCapture(self.url)
  def release(self):
    self.cap.release()
  def detect(self):
    self.results=self.model(self.frame)
  def get_info(self,names):
    self.df=self.results.pandas().xyxy[0]
    if len(names)>0:
        self.df=self.df[self.df['name'].isin(names)]
  def get_info_img(self):
    self.img=self.frame.copy()
    for i in self.df.index:
      p1=(int(self.df.loc[i,'xmin']),int(self.df.loc[i,'ymin']))
      p2=(int(self.df.loc[i,'xmax']),int(self.df.loc[i,'ymax']))
      classname=self.df.loc[i,'name']
      self.img=cv2.rectangle(self.img,p1,p2,(255,0,0),2)
      #self.img=cv2.rectangle(self.img,(p1[0],p1[1]-20),(p2[0],p1[1]),(220,220,220),1)
      self.img=cv2.putText(self.img,classname,(p1[0]+2,p1[1]-5),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,128),2)
        


model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).cpu()
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


names=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 
        'teddy bear', 'hair drier', 'toothbrush']


app = Dash(__name__, external_stylesheets=external_stylesheets)
#app = JupyterDash(__name__)
server = app.server

url='http://181.57.169.89:8080/mjpg/video.mjpg'
obj_detector=yolo_detector(model,url)

obj_detector.capture()
obj_detector.detect()
obj_detector.get_info(names)
obj_detector.get_info_img()
frame=cv2.cvtColor(obj_detector.img,cv2.COLOR_BGR2RGB)
fig=px.imshow(frame)

app.layout = html.Div([
    html.H2('YOLOV5 with Dash'),
    html.P('Choose a class to detect (leave it blank if you want to see all classes)'),
    html.Div(dcc.Dropdown(id='classes',options=[{'label':i,'value':i} for i in names],multi=True)),
    html.Div([
        html.Div(dcc.Graph(id='video-frame',figure=fig),style={'display':'inline-block','width':'60%'}),
        html.Div(dcc.Graph(id='counts-fig'),style={'display':'inline-block','width':'40%'})
    ]),
    html.Div(id='log'),
    dcc.Interval(id='timer',interval=600,n_intervals=0)
   
])


@app.callback(Output('video-frame','figure'),
              Output('log','children'),
              Output('counts-fig','figure'),
             [Input('timer','n_intervals'),
             Input('classes','value')])
def show_video(t,names):
    frame=np.zeros((640,480,3),dtype=np.int8)
    table_fig=go.Figure()
    if names is None:
        names=[]
    try:
        
        start=time.time()
        
        obj_detector.capture()
        
        obj_detector.detect()
        obj_detector.get_info(names)
        obj_detector.get_info_img()
        frame=cv2.cvtColor(obj_detector.img,cv2.COLOR_BGR2RGB)
       
        fig=px.imshow(frame)
        
        df_count=obj_detector.df.groupby(['name']).count()[['class']].reset_index()
        df_count=df_count.rename(columns={'class':'count'})
        
        table_fig.add_trace(go.Table(
            header=dict(values=df_count.columns.to_list(),
                        align='left',
                        line_color='darkslategray',
                        fill_color='#00008B',
                        font=dict(color='white')),
            cells=dict(values=[df_count['name'].to_list(),df_count['count'].to_list()],
                        align='left',
                        line_color='darkslategray',
                        fill_color='white')
           
        ))
        
        
        end=time.time()-start
        return fig,f'yolo detecting in :{round(end,2)} seconds',table_fig
    except Exception as e:
        fig=px.imshow(frame)
        #obj_detector.release()
        #obj_detector=yolo_detector(model,url)
        
        
        return fig,e,table_fig
   

    

if __name__ == '__main__':
    app.run_server(debug=True)
    #app.run_server(mode='inline')