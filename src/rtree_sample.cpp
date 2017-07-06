/*
*name        :rtree_sample.cpp
*auter       :Seiichirou Nomura
*discription :cvRandomTree
*date        :20161208-20170110
*/
#include<iostream>
#include<fstream>
#include<string>
#include<cstring>
#include<stdlib.h>
#include<sstream> //文字ストリーム

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

using namespace std;
using namespace cv;

void count_row_col( const char* , int*, int*);
struct TRAIN_DATA *load_to_train( const char* );
void cvtrees_data_make( const struct TRAIN_DATA* , const char*);
int libsvm_predict( const struct TRAIN_DATA*, const CvRTrees* );

//学習データ保存用の構造体
typedef struct TRAIN_DATA{
  int train_num;//学習に用いるデータ数
  double label;//教師用ラベル
  int data_num;//学習に用いる素性データ数
  double *data;//素性データの配列
}TRAIN_DATA;

/*
*CSVファイルの行と列を数える
*/
void count_row_col( const char *filename , int *row , int *col ){
  int col_num , row_num;

  //ファイルを読み込んで行列を数える
  ifstream ifs(filename);
  row_num = col_num = 0;
  if(!ifs){
    cout<<"入力エラー"<<endl;
  }else{
    string buf;
    while (getline(ifs, buf)) {
      istringstream stream(buf);
      string token;
      if(row_num == 0 ){//カンマ区切りで分割
        while ( getline( stream, token , ','))col_num++;
      }
      row_num++;
    }
  }
  //行列の値をポインタに渡す
  (*col) = col_num;
  (*row) = row_num;
}

/*
*CSVファイルから学習データを取得する
*/
struct TRAIN_DATA *load_to_train( const char *train_filename ){
  int row, col;
  struct TRAIN_DATA *t_data;

  //CSVファイルの行列を取得する
  count_row_col( train_filename , &row , &col );

  //学習に用いるデータを保存する配列の初期化
  t_data = (struct TRAIN_DATA* )malloc( sizeof(struct TRAIN_DATA) * row + 1);
  for( int i = 0 ; i < row ; i++ ){
    t_data[i].train_num = 0;
    t_data[i].label = 0;
    t_data[i].data_num = 0;
    t_data[i].data = (double* )malloc( sizeof(double) * (col - 1));
  }

  //csvファイルを1行ずつ読み込む
  ifstream ifs(train_filename);
  string buf;
  int i = 0 , j = 0;
  while(getline(ifs,buf)){
    t_data[i].train_num = row;
    t_data[i].data_num = col - 1;

    string token;
    istringstream stream(buf);
    //1行のうち、文字列とコンマを分割する
    while(getline(stream,token,',') ){
      double temp = (double)stod(token);
      if( j == 0 ){//行の先頭はラベルとして扱う
        t_data[i].label = temp;
      }else{
        //すべて文字列として読み込まれるためdouble型に変換
        t_data[i].data[j-1] = temp;
      }
      j++;
    }
    i++;
    j = 0;
  }
  return t_data;
}

/*
*特徴量から学習データの作成
*/
void cvtrees_data_make( const struct TRAIN_DATA *t_data , const char *save_filename ){
  CvRTrees model;
  //素性データ数
  int train_num = t_data[0].train_num;
  int data_num = t_data[0].data_num;

  cout << "learning datas :" << train_num << endl;

  Mat training_label( train_num ,1, CV_32FC1 );
  Mat training_data( train_num , data_num , CV_32FC1 );

  for( int i = 0 ; i < train_num; i++){
    //ラベルをふる
    training_label.at<float>(i , 0) = t_data[i].label;
    for( int j = 0; j < data_num; j++){
      training_data.at<float>(i , j) = t_data[i].data[j];
    }
  }

  //学習する
  cout << "Ready to train ..." << endl;
  model.train(training_data,  CV_ROW_SAMPLE , training_label);
  cout << "Finished ..." << endl;

  // 学習結果のファイル出力
  model.save( save_filename );
}

/*
*ランダムフォレストによる検定
*/
int cvtrees_predict( const struct TRAIN_DATA *t_data , const CvRTrees *model ){
  int data_num = t_data->data_num;
  Mat test( data_num , 1 , CV_32FC1 );

  cout << "predict training samples ..." << endl;

  for( int i = 0 ; i < data_num; i++){
    test.at<float>( i , 0 ) = t_data->data[i];
  }

  // opencvによるpredict
  const auto result = static_cast<int>( model->predict( test )  );

  return result;
}

/*
*実行関数
*/
int main(int argc, char* argv[])
{
  char train_filename[248] , save_filename[248];

  //コンソール入力をpathに代入
  strcpy( train_filename, argv[ 1 ] );
  strcpy( save_filename , argv[ 2 ] );

  //CSVファイルから学習データを作成
  struct TRAIN_DATA* t_data;
  t_data = load_to_train( train_filename);

  //学習データの作成
  CvRTrees model;
  cvtrees_data_make( t_data , save_filename);
  model.load( save_filename );

  //テスト用データの作成
  struct TRAIN_DATA* test;
  test = (struct TRAIN_DATA* )malloc( sizeof(struct TRAIN_DATA));
  test->data = (double*)malloc( sizeof(double) * 10);
  test->data_num = 10;
  for(int j = 0; j < 10 ; j++){
    test->data[j] = 0.6;
  }

  //予測
  int result = cvtrees_predict( test , &model );
  cout << "done" << endl;
  cout << "RESULT : correct =" << result << endl;

  // 後始末
  free(t_data);
  free(test);

  return 0;
}
