import os
from flask import Flask, redirect, url_for, send_from_directory,request, render_template
from detectLeukemia import *
import cv2
import matplotlib.pyplot as plt

UPLOAD_FOLDER = 'E:/Aashka/College/SEM5/IP/Mini Project/static/images/user'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'bmp'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

def allowed_file(filename):
    for i in range(len(filename)):
        if filename[i]=='.':
            ext = filename[i+1:]
            break
    if ext in ALLOWED_EXTENSIONS:
        return True

@app.route('/',methods=["GET", "POST"])
def index():
    return render_template('homepage.html')

@app.route('/detectLeukemia',methods=["GET", "POST"])
def detectLeukemia():
    if request.method== 'POST':
        if 'file' not in request.files:
            return render_template('detectLeukemia.html')
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename= file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            a = cv2.imread('static/images/user/{}'.format(filename))
            a = cv2.resize(a,(256,256))
            img1 = cv2.cvtColor(a,cv2.COLOR_BGR2RGB)
            imgcopy = img1.copy()
            img = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            cv2.imwrite(UPLOAD_FOLDER+'/file1.jpg',a)
            cv2.imwrite(UPLOAD_FOLDER+'/file2.jpg',img)
            
            img2 = contrastStretching(img, 20, 150, 0.1, 1, 2)
            cv2.imwrite(UPLOAD_FOLDER+'/file3.jpg',img2)

            imghist = histeq(img)
            cv2.imwrite(UPLOAD_FOLDER+'/file4.jpg',imghist)

            imgfinal = enhancement(img2,imghist)
            cv2.imwrite(UPLOAD_FOLDER+'/file5.jpg',imgfinal)
            imgthresh = thresholding(imgfinal, 150)
            cv2.imwrite(UPLOAD_FOLDER+'/file6.jpg',imgthresh)
            mask = [[1,1,1],
                    [1,0,1],
                    [1,1,1]]
            erodedimg = erosion(imgthresh,mask)
            openedimg = dilation(erodedimg,mask)
            cv2.imwrite(UPLOAD_FOLDER+'/file7.jpg',openedimg)
            imgEdges = edgeDetection(openedimg)
            cv2.imwrite(UPLOAD_FOLDER+'/file8.jpg',imgEdges)
            imgcircle,ctr = detectCircles(img,openedimg)
            cv2.imwrite(UPLOAD_FOLDER+'/file9.jpg',imgcircle)
            return render_template('result.html', cells=ctr)
                #redirect(url_for('detectLeukemia', )
     
    return render_template('detectLeukemia.html')

if __name__ == "__main__":
    app.debug =True
    app.run()
