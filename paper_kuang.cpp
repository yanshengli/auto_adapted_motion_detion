// paper_kuang.cpp : 定义控制台应用程序的入口点。
//

//运动前景检测——基于自适应背景更新

#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

#define threshold1 5 //静态检测的域值
#define Pathfile "video3.avi"
#define SLOTnum 10 //定义的是最小聚类单位
#define SampleNo 50//初始样本个数
#define PI 3.142 
#define threshold2 0.0047//核密度估计判断是否为前景的值
#define threshold3 0.45


////////////////////////////////////////////////////读取视频，传入的参数为视频的位置及名字///////////////////
VideoCapture ReadFile(string pathfile)
{	
	std::string videoFileName = pathfile;
	cv::VideoCapture capture;
	capture.open(videoFileName);
	if (!capture.isOpened())
	{
		std::cout<<"cannot open video"<<std::endl;
		return -1;
	}
	return capture;

}

/////////////////////////////////////////////////////////平均背景法中，求静态平均背景////////////////////////////
Mat init_Backgound(string pathfile)
	{
	VideoCapture capture=ReadFile(pathfile);
	Mat grayImg,grayMat,frame;
	capture.read(frame);	
	int height=frame.rows;
	int width=frame.cols;
	Mat matcount=Mat::zeros(height,width,CV_32FC1);	
	//int frameno= capture.get(CV_CAP_PROP_FRAME_COUNT);
	int frametostart=0;
	int frametoend=SampleNo;
	int currentframe=0;
	while(currentframe<frametoend)
	{

	cv::cvtColor(frame, grayImg, CV_BGR2GRAY);
	grayImg.convertTo(grayMat, CV_32FC1);
	matcount=matcount+grayMat;	
	currentframe++;
	}
	
	CvMat cvMat=matcount;
	CvMat *backMat=cvCreateMat(height,width,CV_32FC1);
	cvZero(backMat);	
	for (int i=0;i<height;i++){		
		for (int j=0;j<width;j++)
		{	
								
			cvmSet(backMat,i,j,(cvmGet(&cvMat,i,j)/double(SampleNo)));
					
		}
	
	}
	
	capture.release();
	
	return backMat;

	}

////////////////////////////////////////////////////////////利用平均背景法，进行前景背景判断/////////////////////////////////////////
int backfilter(Mat init_back,string pathfile)
{
	VideoCapture capture=ReadFile(pathfile);
	Mat frame,grayImg,grayMat,result,resultimg,init_backimg;
	int height=init_back.rows;
	int width=init_back.cols;	
	int framno=0;
	Mat filtermat=Mat::zeros(height,width,CV_32FC1);
	init_back.convertTo(init_backimg,CV_8U);
	CvMat value;
	while (capture.read(frame))
	{	framno++;
		cv::cvtColor(frame, grayImg, CV_BGR2GRAY);
		grayImg.convertTo(grayMat, CV_8U);
		absdiff(init_backimg,grayMat,result);
		result.convertTo(result,CV_32FC1);		
		
		CvMat cvMat=result;
		CvMat cvfiltermat=filtermat;
		for (int i=0;i<height;i++){		
			for (int j=0;j<width;j++)
			{	
				float tmp=cvmGet(&cvMat,i,j);			
				if (tmp>threshold1)
				{	
					
					cvmSet(&cvfiltermat,i,j,255);
				}

				else
				{
					cvmSet(&cvfiltermat,i,j,0);
				}

			}

		}	

		Mat(&cvfiltermat).convertTo(resultimg,CV_8U);
		Mat gaussi_result;
		cv::medianBlur(resultimg,gaussi_result,5);
		//cv::erode(gaussi_result, gaussi_result, cv::Mat());
		// 膨胀
		//cv::dilate(gaussi_result, gaussi_result, cv::Mat());
		cv::imshow("video",frame);
		cv::imshow("foreground", gaussi_result);
		cv::waitKey(10);
		
		}
	capture.release();
	return framno;
		
	}

/////////////////////////////////////////////////////////聚类的函数，输入的为要聚类的样本，每一行存一帧的所有像素，输出的为每一行所属的类别。。。0或1
CvMat* GrayImageSegmentByKMeans2(CvMat *samples)
{

	//CvMat*samples = cvCreateMat((pImg->width)* (pImg->height),sampleno, CV_32FC1);
	//创建类别标记矩阵，CV_32SF1代表位整型通道
	//cout<<samples->width<<endl<<samples->height<<endl;
	CvMat *clusters = cvCreateMat(SLOTnum,1, CV_32SC1);
	//创建类别中心矩阵
	int nClusters=2;
	//CvMat *centers = cvCreateMat(nClusters, sampleno, CV_32FC1);
	// 将原始图像转换到样本矩阵	
	
	
	//开始聚类，迭代次，终止误差.0
	cvKMeans2(samples, nClusters,clusters, cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS,100, 1.0),3,0);	
	//cout<<Mat(clusters)<<endl;
	//cout<<clusters->width<<clusters->height<<endl;

	return clusters;
}    


//这个函数就是得到10帧里面的聚类之后的关键的两帧，而且是对像帧的像素进行聚类，最后结果返回getkeyimage，它完成了keyframe和GrayImageSegmentByKMeans2两个函数的功能
CvMat* GrayImageSegmentByKMeans1(CvMat *samples)
{

	//CvMat*samples = cvCreateMat((pImg->width)* (pImg->height),sampleno, CV_32FC1);
	//创建类别标记矩阵，CV_32SF1代表位整型通道
	//cout<<samples->width<<endl<<samples->height<<endl;
	CvMat *clusters = cvCreateMat(SLOTnum,1, CV_32SC1);
	CvMat *sample=cvCreateMat(SLOTnum,1,CV_32FC1);
	CvMat *keyframe = cvCreateMat(2,samples->width+1, CV_32FC1);
	CvMat *cluster=cvCreateMat(SLOTnum,1,CV_32FC1);
	cvZero(keyframe);
	//创建类别中心矩阵
	int nClusters=2;
	int class0=0,class1=0;
	//CvMat *centers = cvCreateMat(nClusters, sampleno, CV_32FC1);
	// 将原始图像转换到样本矩阵	

	for(int k=0;k<samples->cols;k++)
	{	int j=0;
		class0=0;
		class1=0;
		while(j<SLOTnum)
		{
			cvmSet(sample,j,0,(cvmGet(samples,j,k)));
			j++;
		}
		cvKMeans2(sample, nClusters,clusters, cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS,100, 1.0),3,0);
		
		cvConvert(clusters,cluster);
		for (int i=0;i<SLOTnum;i++){
		if (int(cvmGet(cluster,i,0))==0)
		{
			class0++;		
			double tmp = cvmGet(sample,i,0);
			//cvSet2D(samples,p,k++, s);
			cvmSet(keyframe,0,k,(tmp+cvmGet(keyframe,0,k)));
		

		}
		else
			{
			class1++;
				
			double tmp = cvmGet(sample,i,0);
		//cvSet2D(samples,p,k++, s);
			cvmSet(keyframe,1,k,(tmp+cvmGet(keyframe,1,k)));

			}
		}
		cvmSet(keyframe,0,k,cvmGet(keyframe,0,k)/float(class0));
		cvmSet(keyframe,1,k,cvmGet(keyframe,0,k)/float(class1));	
} 
	cvmSet(keyframe,0,samples->width,0.1);
	cvmSet(keyframe,1,samples->width,0.1);
	return keyframe;
}

//////////////////////////////////////////此通用函数，从输入的样本帧中，得到聚类之后的关键帧，并统计其权重,输出带有权重的关键帧，行存储//////////////////////////////
CvMat *keyframe(CvMat * clusterMat,CvMat * samples,int row,int coloum)//统计聚类之后所占的比例
{
int class0=0,class1=0;
float weight_class0;

CvMat* keyframe = cvCreateMat(2,(row)* (coloum)+1, CV_32FC1);
cvZero(keyframe);
CvMat *cluster=cvCreateMat(SLOTnum,1,CV_32FC1);
cvConvert(clusterMat,cluster);
//Mat(clusterMat).convertTo(cluster, CV_32FC1);
for (int i=0;i<SLOTnum;i++)
{

if (int(cvmGet(cluster,i,0))==0)
{
	class0++;
		
		for(int j=0;j < coloum*row; j++)
		{	
			double tmp = cvmGet(samples,i,j);
			//cvSet2D(samples,p,k++, s);
			cvmSet(keyframe,0,j,(tmp+cvmGet(keyframe,0,j)));
		}
	
}
else
{
	class1++;
	for(int j=0;j < coloum*row; j++)
	{	
		double tmp = cvmGet(samples,i,j);
		//cvSet2D(samples,p,k++, s);		
		cvmSet(keyframe,1,j,(tmp+cvmGet(keyframe,1,j)));
	}

}
}

for(int j=0;j < coloum*row; j++)
{	
	
	cvmSet(keyframe,0,j,cvmGet(keyframe,0,j)/float(class0));
	cvmSet(keyframe,1,j,cvmGet(keyframe,0,j)/float(class1));
}

weight_class0=float(class0)/float(SLOTnum);
cvmSet(keyframe,0,coloum*row,weight_class0/(SampleNo/SLOTnum));///(SampleNo/SLOTnum)//
cvmSet(keyframe,1,coloum*row,(1-weight_class0)/(SampleNo/SLOTnum));///(SampleNo/SLOTnum)//
return keyframe;

}

//////////////////////////////////////////////////////////////////得到最终所有样本的关键帧及其权重矩阵，即50个样本中得到10个关键帧，它要调用keyframe和GrayImageSegmentByKMeans2函数
////同时也可以调用GrayImageSegmentByKMeans1函数，完成对像素级的聚类，返回2帧关键帧/////////////////////////
CvMat* getkeyimage( string pathfile,int row,int coloum)
{
VideoCapture capture=ReadFile(pathfile);

int frameToStart=0;
int frameToEnd=50;
int totalFrame=capture.get(CV_CAP_PROP_FRAME_COUNT);
Mat frame,grayMat;
CvMat * keymat=cvCreateMat(2*SampleNo/SLOTnum,(row)* (coloum)+1, CV_32FC1);
int currentframe=frameToStart;
CvMat*samples = cvCreateMat(SLOTnum,(row)* (coloum), CV_32FC1);
int p = 0;
CvMat *clusterMat;
CvMat *resultMat;
int line=0;
while(currentframe<frameToEnd)  
{  
	int k=0;
	//读取下一帧  
	if(!capture.read(frame))  
	{  
		cout<<"读取视频失败"<<endl;  
		
	} 
	cv::cvtColor(frame, grayMat, CV_BGR2GRAY);
	grayMat.convertTo(grayMat, CV_32FC1);
	CvMat gray=grayMat;
	CvScalar s;
	for(int i = 0; i < row; i++)
	{
		for(int j=0;j < coloum; j++)
		{	
			s.val[0] = cvmGet(&gray,i,j);
			//cvSet2D(samples,p,k++, s);
			cvmSet(samples,p,k++,s.val[0]);
		}
	}
	currentframe++;
	p++;
	if (currentframe%SLOTnum==0)
	{	p=0;
		
		clusterMat=GrayImageSegmentByKMeans2(samples);
		resultMat=keyframe(clusterMat,samples,row,coloum);//得到关键帧及其比例
		//resultMat=GrayImageSegmentByKMeans1(samples);
		for(int i=0;i<row*coloum+1;i++)
		{
			double tmp0 = cvmGet(resultMat,0,i);
			double tmp1 = cvmGet(resultMat,1,i);
			cvmSet(keymat,line,i,tmp0);
			cvmSet(keymat,line+1,i,tmp1);
		}
				
		line=line+2;
	}
	
}  
capture.release();
return keymat;

}

//////////////////////////////////////////////////////////尝试每次都根据关键帧跟新中位数，每计算一次，更新一次，但是这样来做的话，费时间/////////////////////////////////
CvMat * MedianNo_1(CvMat *keyframe)
{

	Mat frame,framemat;
	CvMat *mdianmat=cvCreateMat(SLOTnum-1,keyframe->width-1, CV_32FC1);
	int i=0;
	while(i<SLOTnum-1)
	{	for(int j=0;j<(keyframe->width-1);j++)
	{
		double temp=abs(cvmGet(keyframe,i,j)-cvmGet(keyframe,i+1,j));
		cvmSet(mdianmat,i,j,temp);
	}
	i++;
	}

	//CvMat *medianResult=cvCreateMat(rows*coloums,SampleNo, CV_32FC1);
	Mat medianResult;
	sortIdx(Mat(mdianmat),medianResult,CV_SORT_EVERY_COLUMN+CV_SORT_DESCENDING);

	CvMat *result=cvCreateMat(1,keyframe->width-1, CV_32FC1);

	CvMat median=medianResult;
	CvMat *median_float=cvCreateMat(SLOTnum-1,keyframe->width-1, CV_32FC1);
	cvConvert(&median,median_float);

	for (int i=0;i<keyframe->width-1;i++)
	{
		double temp1 =cvmGet(median_float,SLOTnum/2-1,i);		

		cvmSet(result,0,i,temp1);

	}
	return result;

}
////////////////////////////////////求中位数，每个像素的中位数，一行存储,前50个样本的中位数/////////////////////////////
CvMat * MedianNo(int framstart,int frameend,VideoCapture capture,int rows,int coloums)
{
Mat frame,framemat;
Mat previousframe,previousmat;
Mat resultmat,medianResult;
int p=0;
capture.read(previousframe);
cv::cvtColor(previousframe, previousmat, CV_BGR2GRAY);
previousmat.convertTo(previousmat, CV_8U);
CvMat *mdianmat=cvCreateMat(rows*coloums,SampleNo-1, CV_32FC1);
while(framstart<frameend-1)
{	int k=0;
	capture.read(frame);
	cv::cvtColor(frame, framemat, CV_BGR2GRAY);
	framemat.convertTo(framemat, CV_8U);
	absdiff(framemat,previousmat,resultmat);
	resultmat.convertTo(resultmat, CV_32FC1);	
	
	CvMat result=resultmat;
	for(int i = 0; i < rows; i++)
	{
		for(int j=0;j < coloums; j++)
		{	
			double temp = cvmGet(&result,i,j);			
			cvmSet(mdianmat,k++,p,temp);
		}
	}
	p++;
	framstart++;
	
}

//CvMat *medianResult=cvCreateMat(rows*coloums,SampleNo, CV_32FC1);
sortIdx(Mat(mdianmat),medianResult,CV_SORT_EVERY_ROW+CV_SORT_DESCENDING);

CvMat *result1=cvCreateMat(1,rows*coloums, CV_32FC1);

CvMat median=medianResult;

CvMat *cluster=cvCreateMat(rows*coloums,SampleNo-1, CV_32FC1);
cvConvert(&median,cluster);

for (int i=0;i<rows*coloums;i++)
{
	double temp1 =cvmGet(cluster,i,SampleNo/2-1);	
	
	cvmSet(result1,0,i,temp1);

}
return result1;

}

//////////计算核密度估计表达式的函数，返回最后得到的结果，一行存储////////////////////////////////
CvMat * calculate_KDE(Mat frame,CvMat *keyframe,CvMat * medianNumber)
{
CvMat frammat=frame;
double coe;//计算指数上的系数
double weight;//计算指数前面的比重系数
double kderesult;//最后得到的每一项的核密度值
double h;//窗口大小
double count=0;//计算的最终的某一个像素的概率值
CvMat *frameline=cvCreateMat(1, frame.rows*frame.cols,CV_32FC1);//把二维帧转换为一维矩阵
CvMat *kde_result=cvCreateMat(1, frame.rows*frame.cols,CV_32FC1);
int k=0;
for(int i = 0; i < frame.rows; i++)
{
	for(int j=0;j < frame.cols; j++)
	{	
		double temp = cvmGet(&frammat,i,j);			
		cvmSet(frameline,0,k++,temp);
	}
}
//所有的数据，全部都转换为一维的，即一行数据代表一个帧信息 

for(int j=0;j<frame.rows *frame.cols;j++)
{	count=0;
	for(int i=0;i<2*SampleNo/SLOTnum;i++)
	{	double h=(cvmGet(medianNumber,0,j)/0.962);
		coe=abs(cvmGet(frameline,0,j)-cvmGet(keyframe,i,j))*abs(cvmGet(frameline,0,j)-cvmGet(keyframe,i,j));
		coe=-0.5*coe/(h*h);
		weight=(cvmGet(keyframe,i,frame.rows*frame.cols))/sqrt(2*PI*h*h);
		kderesult=exp(coe)*weight;
		count=count+kderesult;
	}
	
	cvmSet(kde_result,0,j,count);
}

return kde_result;
}

//////背景更新算法，每算出一帧，更新样本，按照先进后出的原则/////////////////////////
CvMat * Updatebackground_kde(CvMat * backgroundresult,CvMat *keyframe)
{

	for (int i=0;i<backgroundresult->width;i++)
	{
		double temp=cvmGet(backgroundresult,0,i);

		if(temp>threshold2)
		{
			cvmSet(keyframe,0,i,temp);

		}
	}

	CvMat *temp=cvCreateMat(1,keyframe->width,CV_32FC1);
	for (int i=0;i<keyframe->width;i++)
	{
		cvmSet(temp,0,i,cvmGet(keyframe,0,i));

		for (int j=0;j<SLOTnum-1;j++)
		{
			cvmSet(keyframe,j,i,cvmGet(keyframe,j+1,i));
		}

		cvmSet(keyframe,SLOTnum-1,i,cvmGet(keyframe,0,i));
	}

	return keyframe;

}


/////始终只更新第0行的数据
CvMat * Static_Updatebackground_kde(CvMat * backgroundresult,CvMat *keyframe)
{

	for (int i=0;i<backgroundresult->width;i++)
	{
		double temp=cvmGet(backgroundresult,0,i);

		if(temp>threshold2)
		{
			cvmSet(keyframe,9,i,temp);

		}
	}
	
	return keyframe;

}

///////////////////////////////////////背景更新算法，突变更新，每得到10个样本后，聚类得到2个关键帧，进行替换，替换第0，第1行的数据//////////
CvMat *keyfrUpdatebackground_cluster(CvMat *frame_10,CvMat *keymat,int rows,int coloums)
{

CvMat *clusterMat;
CvMat *resultMat;
clusterMat=GrayImageSegmentByKMeans2(frame_10);
resultMat=keyframe(clusterMat,frame_10,rows,coloums);//得到关键帧及其比例
//resultMat=GrayImageSegmentByKMeans1(frame_10);
int c_1=0;
int c_2=1;
for(int i=0;i<rows*coloums;i++)
{
	double tmp0 = cvmGet(resultMat,0,i);
	double tmp1 = cvmGet(resultMat,1,i);		
	cvmSet(keymat,c_1,i,tmp0);
	cvmSet(keymat,c_2,i,tmp1);
}
double tmp2 = cvmGet(resultMat,0,rows*coloums);
double tmp3 = cvmGet(resultMat,1,rows*coloums);		
cvmSet(keymat,c_1,rows*coloums,tmp2/3);
cvmSet(keymat,c_2,rows*coloums,tmp3/3);
return keymat;

}


////去燥音函数，寻找轮廓，计算其面积，将面积较小的中心为白色点的轮廓去掉，最后得到轮廓比较大的，去掉小的燥声///////////////
CvMat * removenoise(CvMat frame)
{
	CvMemStorage* storage = cvCreateMemStorage(0);
	//Mat frame_src;
	Mat frame_src(frame.rows,frame.cols,CV_8U);
	Mat(&frame).convertTo(frame_src,CV_8U);
	CvContourScanner scanner = NULL;     
	//IplImage* img_Clone=cvCloneImage(Mat(&frame))
	IplImage  img_Clone= IplImage(frame_src); 
	scanner = cvStartFindContours(&img_Clone,storage,sizeof(CvContour));     
	//开始遍历轮廓树     
	CvRect rect;
	CvSeq * contour =NULL;
	uchar *pp;
	while (contour=cvFindNextContour(scanner))     
	{     
		double tmparea = fabs(cvContourArea(contour));     
		rect = cvBoundingRect(contour,0);        
		if (tmparea <10)     
		{     
						//当连通域的中心点为黑色时，而且面积较小则用白色进行填充     
			pp=(uchar*)(img_Clone.imageData + img_Clone.widthStep*(rect.y+rect.height/2)+rect.x+rect.width/2);     
			if (pp[0]==255)     
			{     
				for(int y = rect.y;y<rect.y+rect.height;y++)     
				{     
					for(int x =rect.x;x<rect.x+rect.width;x++)     
					{     
						pp=(uchar*)(img_Clone.imageData + img_Clone.widthStep*y+x);     

						if (pp[0]==0)     
						{     
							*(img_Clone.imageData + img_Clone.widthStep*y+x)=0;     
						}     
					}     
				}     
			}     

		}     
	} 
	CvMat *mat = cvCreateMat( frame.rows, frame.cols, CV_32FC1);
	cvConvert( &img_Clone, mat );
	return mat;
}


/*阴影检测，这个主要是转换为HSV空间，然后根据空间的亮度分量进行进行计算，具体是核密度结果矩阵一行存储，以及原始的帧，将原始帧进行空间转换，得到满足要求的相应点的坐标，把
相应结果帧中该坐标点的位置值置为0
*/
CvMat * detect_hide(Mat currentFrame,CvMat* pFrImg)
{//	cout<<pFrImg->width<<endl;
	IplImage *HSVImg1 = cvCreateImage(cvSize(currentFrame.cols,currentFrame.rows), IPL_DEPTH_32F, 3);
	IplImage *HSVImg=cvCreateImage(cvSize(currentFrame.cols,currentFrame.rows),IPL_DEPTH_32F,3);
	//cout<<currentFrame.channels();
	//cout<<currentFrame.depth();
	//CvMat curr_frame=currentFrame;
	IplImage curr_frame=IplImage(currentFrame);
	//cout<<curr_frame.type<<endl;
	cvConvertScale(&curr_frame,HSVImg1,1.0/255);
	cvCvtColor(HSVImg1, HSVImg, CV_RGB2HSV);
	//cout<<HSVImg->imageData[0]<<endl;
	float total_y_v = 0.0;
	//   int y = 0;
	for ( int y = 0; y < HSVImg->height; y++) {
		float total_v = 0.0;

		for (int x = 0; x < HSVImg->width; x++) {
			int n = HSVImg->width * y + x;
			int v = HSVImg->imageData[n * 3 + 2];
			//cout<<v<<endl;
			total_v +=(v+256)%256;
		}
		total_y_v += (float)total_v / HSVImg->width;
	}
	float avg_v = total_y_v / HSVImg->height;//求出平均亮度
	//cout<<avg_v<<endl;
	for (int y = 0; y < HSVImg->height; y++) 
	{
		for (int x = 0; x < HSVImg->width; x++) 
		{
			int n = HSVImg->width * y + x;
			if (cvmGet(pFrImg,y,x)!= 0) 
			{//	cout<<n<<endl;
				//int h = HSVImg->imageData[n * 3];
				//int s = HSVImg->imageData[n * 3 + 1];
				int v = HSVImg->imageData[n * 3 + 2];
				//cout<<h<<endl/*;*/
				if (((v+256)%256)<avg_v/3) 
				{
					cvmSet(pFrImg,y,x,0);
				}
			}
		}
	}
	cvReleaseImage(&HSVImg);
	return pFrImg;
}

////显示最后得到的结果，以灰度图的形式，这里面包括去燥声函数的调用，阴影检测的调用，平滑滤波，中值滤波，腐蚀，膨胀//////////////////////
int display(CvMat framemat,CvMat *result,Mat frame_time,Mat current_frame)
{
//把result的结果变成多维的,这样好用于显示。
CvMat *precision_result=cvCreateMat(framemat.rows,framemat.cols,CV_32FC1);
CvMat *detect_result=cvCreateMat(1,framemat.rows*framemat.cols,CV_32FC1);
cvZero(precision_result);
Mat resultimg;
int row=0;
int coloum=0;

for(int i=0;i<framemat.rows*framemat.cols;i++)
{
if(i%framemat.cols==0 &&i!=0)
	{
	row++;
	coloum=0;
	}
	cvmSet(precision_result,row,coloum,cvmGet(result,0,i));
	coloum++;
}

//CvMat * frame_shade_result=detect_shade(framemat,frame_time,precision_result);

CvMat frame=framemat;

//cout<<Mat(precision_result)<<endl;
for(int i=0;i<framemat.rows;i++)
	for(int j=0;j<framemat.cols;j++)
	{
		if(cvmGet(precision_result,i,j)>threshold2)
		{
			cvmSet(&frame,i,j,0);
		}
		else
			cvmSet(&frame,i,j,255);

	}
	
	//detect_result=detect_hide(current_frame,&frame);
	//CvMat * frame_noise=removenoise(frame);
	//cvSmooth(&frame,&frame, CV_GAUSSIAN, 3, 0, 0);
	Mat(&frame).convertTo(resultimg,CV_8U);		
	Mat gaussi_result;
	cv::medianBlur(resultimg,gaussi_result,7);

	//cv::blur(resultimg,gaussi_result,cv::Size(5,5));
	//cv::medianBlur(resultimg,gaussi_result,7);//中值滤波去澡

	//cv::erode(resultimg, resultimg, cv::Mat());//形态学修正，腐蚀，膨胀
	// 膨胀
	//cv::dilate(resultimg, resultimg, cv::Mat());	
	cv::imshow("foreground", Mat(gaussi_result));
	cv::waitKey(10);
	return 0;

}

///该函数计算突变的像素点的个数，把相邻两帧相减，看不为0的像素的个数////
int cout_change(Mat frame)
{
int count=0;
frame.convertTo(frame,CV_32FC1);
CvMat framemat=frame;
for(int i=0;i<frame.rows;i++)
	for(int j=0;j<frame.cols;j++)
		if (cvmGet(&framemat,i,j)>0)
			count++;
return count;

}

///突变更新，把10帧中随机选择一帧，把它完全替换/////
CvMat *change_update(CvMat *keyframe,CvMat cvframe)
{	int k=0;
	int c_1=rand()%10;
	for(int i = 0; i < cvframe.rows; i++)
	{	
		for(int j=0;j < cvframe.cols; j++)
		{	
			double temp = cvmGet(&cvframe,i,j);			
			cvmSet(keyframe,c_1,k++,temp);
		}
	}

	return keyframe;

}
////计算核密度估计的总函数，从50帧开始计算，最后显示，背景更新/////////////////////////////////////////////////
int  KDE(CvMat *keyframe,string pathfile,int rows,int coloums,Mat back_sample)
{
	VideoCapture capture=ReadFile(pathfile);
	int totalFrame=capture.get(CV_CAP_PROP_FRAME_COUNT);		
	Mat frame,framemat,previousframe,previousframemat,frame_sub;
	Mat sub_frame;
	int frameToStart=0;
	int frameToEnd=SampleNo;

	int currentframe=frameToEnd;	
	CvMat *MedianNumber;
	CvMat *frame_10=cvCreateMat(10,rows*coloums,CV_32FC1);//存储10帧	
	MedianNumber=MedianNo(frameToStart,frameToEnd,capture,rows,coloums);
	int frameno=0;
	capture.read(previousframe);
	cv::cvtColor(previousframe, previousframemat, CV_BGR2GRAY);
	//求得背景样本及其权值,存在keyframe中，求得中位数，存储在MedianNumber中，下面就是计算新来样本的概率分布了。
	while(currentframe<totalFrame-1)
	{
	int k=0;	
	capture.read(frame);
	cv::cvtColor(frame, framemat, CV_BGR2GRAY);
	framemat.convertTo(framemat, CV_32FC1);
	CvMat cvframe=framemat;
	for(int i = 0; i < frame.rows; i++)
	{
		for(int j=0;j < frame.cols; j++)
		{	
			double temp = cvmGet(&cvframe,i,j);			
			cvmSet(frame_10,frameno,k++,temp);
		}
	}
		
	CvMat * result=calculate_KDE(framemat,keyframe,MedianNumber);		
	framemat.convertTo(frame_sub,CV_8U);
	previousframemat.convertTo(previousframemat,CV_8U);
	absdiff(frame_sub,previousframemat,sub_frame);
	int changeNO=cout_change(sub_frame);//计算突变的个数

	cv::imshow("video", frame);//显示原来的图像
	display(cvframe,result,back_sample,frame);	
	currentframe++;
	previousframe=frame;
	cv::cvtColor(previousframe, previousframemat, CV_BGR2GRAY);
	frameno++;
	if((currentframe-frameToEnd)%10==0 &&currentframe!=SampleNo)
		{
		//
		//keyframe=keyfrUpdatebackground_cluster(frame_10,keyframe,framemat.rows,frame.cols);//每10帧更新一下
		frameno=0;
		}
	else
		//keyframe=Updatebackground_kde(result,keyframe);
		keyframe=Static_Updatebackground_kde(result,keyframe);//每帧都更新，在整数帧时不更新
		

	if(changeNO>rows*coloums/2)//突变更新
	{
	//keyframe=change_update(keyframe,cvframe);
	}

	}
	capture.release();
	return 0;

}

//////总控制函数，有静态和动态两种///////////////////////////////////////////////////////////

/*
int main(int argc, char** argv)
{	
	cv::Mat back;	
	CvMat *keyframe;
	CvMat *result;
	/////////////////////////////静态处/////////////////////////////////////////////////
	back=init_Backgound(Pathfile);//计算初始背景
	//int frameno=backfilter(back,Pathfile);//计算背景样本，显示静态方法的效果
	/////////////////////////////////////////动态处理/////////////////////////////////////
	keyframe=getkeyimage(Pathfile,back.rows,back.cols);	
	KDE(keyframe,Pathfile,back.rows,back.cols,back);	
	cout<<"视频已分析完"<<endl;
	return 0;
}



*/