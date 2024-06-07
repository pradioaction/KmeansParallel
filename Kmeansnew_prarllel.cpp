#include <ctime>     // for a random seed
#include <fstream>   // for file-reading
#include <iostream>  // for file-reading
#include <sstream>   // for file-reading
#include <vector>
#include <fstream>
#include <memory>
#include <cmath>
#include <algorithm>
#include "omp.h"
using namespace std;

int k = 1000;
int tactics = 1;
int minP = 0;
int maxP = 10000;

struct Point {
    double x, y;     // coordinates
    int cluster;     // no default cluster
    double minDist;  // default infinite dist to nearest cluster
    // static int K;
    // static int tactics;
    Point() : 
        x(0.0), 
        y(0.0),
        cluster(-1),
        minDist(__DBL_MAX__) {}
        
    Point(double x, double y) : 
        x(x), 
        y(y),
        cluster(-1),
        minDist(__DBL_MAX__) {}

    double distance(Point p) {
        double distance = 0;
        distance = (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y);
        return distance;
    }
};


bool cmpX(const Point &p1, const Point &p2);
bool cmpY(const Point &p1, const Point &p2);

void fileSave(vector<Point>& points,const char* filename);
void kMeansClustering(vector<Point>& points, int epochs, const int threshold);
vector<Point> readPoint(const char* filename, int pointNum);
void centroSet(vector<Point>& points, vector<Point>& centroids);
void renewCentro(vector<int>& nPoints,
                vector<Point>& points,
                vector<double>& sumX,
                vector<double>& sumY,
                vector<Point>& centroids
                );
void deleCluster(vector<int>& nPoints,
                vector<Point>& points,
                vector<Point>& centroids,
                const int threshold
                );
template <typename T>
void input(T &t)
{
	char c;
	cin.get(c);
	if (c != '\n')
	{
		cin.putback(c);
		cin >> t;
		cin.get();    //清楚输入后留下的回车，或者直接清空缓存区也可
	}
}


int main() {
    const char* filename = "./gattempR_SparseSample1D240Folds5200Offset1Layer-NormalResolution.dat";
    // const char* filename = "./gattempS_360D240Folds5200mOffset1Layer.dat";
    // int k = 3;
    // int tactics = 1;
    int threshold = 20;
    int pointNum = -1;
    int  epochs = 10;
    cout << "Enter the value of K (Press Enter for default value 1000): ";
    input<int>(k);
    cout << "Enter the value of epochs (Press Enter for default value 10): ";
    input<int>(epochs);
    cout << " \n[1] for clustering along the x-axis\n[2] for clustering along the y-axis\n[3] for clustering along both x and y axes\nEnter the value of tactics (The default value is 1, Press Enter for default): ";
    input<int>(tactics);
    cout << "If the number of points in the cluster is less than the critical value, the cluster is deleted.\nEnter the value of threshold (The default value is 20, Press Enter for default): ";
    input<int>(threshold);
    cout << "The minimum number of read points.\nEnter the value of pointnum (The default value is all, Press Enter for default): ";
    input<int>(pointNum);
    // void kMeansClustering(vector<Point>& points, int epochs,const int k,const int tactics,const int threshold);
    // vector<Point> readPoint(const char* filename, int pointNum,const int tactics);

    vector<Point> points = readPoint(filename,pointNum);

    kMeansClustering(points,epochs, threshold);
}


bool cmpX(const Point &p1, const Point &p2){
    if (p1.x<p2.x)
    {
        return true;
    }
    else
        return false;
}
bool cmpY(const Point &p1, const Point &p2){
    if (p1.y<p2.y)
    {
        return true;
    }
    else
        return false;
}

vector<Point> readPoint(const char* filename, int pointNum)  {

    ifstream file(filename, ios::in | ios::binary);
    if (!file)
	{
		exit(1);
	}
    //移动到末尾
    file.seekg(0,ios::end);
    //获取点的个数
    std::streampos fileSize = file.tellg(); 
    const size_t unitSize = sizeof(double)*2; // 每个单位的大小
    cout << "MaxPoints:" << (fileSize / unitSize) <<endl;
    size_t numUnits = (pointNum > 0) ? pointNum : (fileSize / unitSize); // 计算单位总数
    // size_t numUnits = 100000000 ; // 计算单位总数
    int numThreads = omp_get_max_threads(); // 获取最大线程数
    // int numThreads = 2; // 获取最大线程数
    size_t unitsPerThread = numUnits / numThreads + 1;
    cout << "numThreads:" << numThreads <<endl;
    file.close();
    vector<Point> points(numUnits);
    #pragma omp parallel 
    {
        int threadId = omp_get_thread_num();
        // std::cout << threadId << std::endl;
        size_t startUnit = threadId * unitsPerThread;
        size_t endUnit = (threadId == numThreads - 1) ? numUnits : (startUnit + unitsPerThread);

        std::ifstream localFile(filename, std::ios::binary);
        localFile.seekg(startUnit * unitSize, std::ios::beg);

        std::vector<std::vector<double>> localPoints;
        double local_xy[2];

        for (size_t i = startUnit; i < endUnit&&(!localFile.eof()); ++i) {
            if (!localFile.read((char*)local_xy, unitSize)) {
                break;
            }
            points[i] = Point(local_xy[0], local_xy[1]);
        }
        localFile.close();
    }
    // std::cout << "Readover" << std::endl;
    std::cout << "Read " << points.size() << " points." << std::endl;
    // Kmeans(points);
    // std::cout << "over" << std::endl;

    return points;
}

void centroSet(vector<Point>& points,vector<Point>& centroids){
    // 迭代点以将数据附加到质心
    for (vector<Point>::iterator c = begin(centroids); c != end(centroids); ++c) {
        // 快速获取簇索引
        int clusterId = c - begin(centroids);
        #pragma omp parallel for
        for (int i = 0; i < points.size(); ++i) {
            Point p = points[i];
            double dist = c->distance(p);
            if (dist < p.minDist) {
                p.minDist = dist;
                p.cluster = clusterId;
            }
            points[i] = p;
        }
    }
}
void fileSave(vector<Point>& points,const char* filename){
    ofstream myfile;
    myfile.open(filename);
    myfile << "x,y,c" << endl;

    // #pragma omp parallel for 
    for (size_t i = 0; i < points.size(); ++i) {
        myfile << points[i].x << "," << points[i].y << "," << points[i].cluster << endl;
    }
    myfile.close();
}

void deleCluster(vector<int>& nPoints,
                vector<Point>& points,
                vector<Point>& centroids,
                const int threshold
                ){

    int count = 0;
    int pointNum = 0;
    #pragma omp parallel for reduction(+:count,pointNum)
    for (int i = 0;i<nPoints.size();i++)
    {
        pointNum+=nPoints[i];
        if(nPoints[i]<threshold){
            // cout<<"cluster "<<i<<" were clean"<<endl;
            count++;
            centroids[i].x=-1;
            centroids[i].y=-1;
            centroids[i].cluster=-1;
            #pragma omp parallel for
            for(int j = 0;j<points.size();j++){
                if(points[j].cluster==i){
                    points[j].cluster=-1;
                }
            }
        }
    }
    cout<<"clusteres "<<" have "<<pointNum<<" points "<<endl;
    cout<<count<<" cluster "<<" were clean"<<endl;
}


void renewCentro(vector<int>& nPoints,
                vector<Point>& points,
                vector<double>& sumX,
                vector<double>& sumY,
                vector<Point>& centroids
                ){

    // 用零初始化
    #pragma omp parallel for
    for (int j = 0; j < k; ++j) {
        nPoints[j] = 0;
        sumX[j] = 0.0;
        sumY[j] = 0.0;
    } 
    //数据准备
    #pragma omp parallel for
    for (int i = 0; i < points.size(); ++i) {
        int clusterId = points[i].cluster;
        #pragma omp atomic
        nPoints[clusterId] += 1;
        #pragma omp atomic
        sumX[clusterId] += points[i].x;
        #pragma omp atomic
        sumY[clusterId] += points[i].y;
        points[i].minDist = __DBL_MAX__;  // 重置距离
    }
    // 计算新的质心
    #pragma omp parallel for
    for (int i = 0; i < centroids.size(); ++i) {
        // int clusterId = i;
        centroids[i].x = sumX[i] / nPoints[i];
        centroids[i].y = sumY[i] / nPoints[i];
        switch(tactics){
            case 1: centroids[i].x = centroids[i].x - (int(centroids[i].x)%150) +( int(centroids[i].x)%150>75 ? 150 :0);
                    // if(minP> centroids[i].x){
                    //     centroids[i].x = minP;
                    // }
                    // if(maxP< centroids[i].x){
                    //     centroids[i].x = maxP;
                    // }
                    break;
            case 2: centroids[i].y= centroids[i].y - (int(centroids[i].y)%150) + ( int(centroids[i].y)%150>75 ? 150 :0);
                    // if(minP> centroids[i].y){
                    //     centroids[i].y = minP;
                    // }
                    // if(maxP< centroids[i].y){
                    //     centroids[i].y = maxP;
                    // }
                    break;
            case 3:
                    centroids[i].x = centroids[i].x - (int(centroids[i].x)%150) +( int(centroids[i].x)%150>75 ? 150 :0);
                    centroids[i].y= centroids[i].y - (int(centroids[i].y)%150) + ( int(centroids[i].y)%150>75 ? 150 :0);
                    break;
            default : break;
        }
    }
}


void kMeansClustering(vector<Point>& points, int epochs,const int threshold){
    //初始化簇集
    vector<Point> centroids;
    int n = points.size();
    srand(0);
    //初始化质心
    for (int i = 0; i < k; ++i) {
        centroids.push_back(points[rand() % n]);
    }
    int count = 0;
    //初始化簇集坐标之和
    vector<int> nPoints(k);
    vector<double> sumX(k), sumY(k);

    while (((++count)<=epochs))
    {
        //更新簇集 设置点所属的簇集
        centroSet(points,centroids);
        //更新簇集质心
        renewCentro(nPoints,points,sumX,sumY,centroids);
        
        //输出该次迭代信息
        cout<< "------------------------the " << count << " iteration------------------------" <<endl;
        for (int i = 0; i < k; i++)   
        {
            cout<< "the cluster " << i << " | the X:" << centroids[i].x << " | the Y:" << centroids[i].y << " have " << nPoints[i] << " points" <<endl;
        }
        cout<< "------------------------the " << count << " iteration------------------------" <<endl;
    }

    cout<< "---------------------the final " << count-1 << " iteration---------------------" <<endl;
    //输出最后迭代信息
    for (int i = 0; i < k; i++)
    {
        cout<< "the cluster " << i << " | the X:" << centroids[i].x << " | the Y:" << centroids[i].y << " have " << nPoints[i] << " points" <<endl;
    }
    cout<< "---------------------the final " << count-1 << " iteration---------------------" <<endl;
    // const char* filename1 = "output_point_prallel.csv";
    const char* filename2 = "output_centro_prallel.csv";
    deleCluster(nPoints,points,centroids,threshold);
    // fileSave(points,filename1);
    fileSave(centroids,filename2);
}