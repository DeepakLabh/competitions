//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h> // mac os x
#include <iostream>
#include <jsoncpp/json/json.h>
#include<jsoncpp/json/writer.h>
#include <fstream>

using namespace std;

const long long max_size = 2000;         // max length of strings
const long long N = 40;                  // number of closest words that will be shown
const long long max_w = 50;              // max length of vocabulary entries

int main(int argc, char **argv) {
//////////////////JSON WRITE CHECK//////////////////////////////////////////////////////
//    Json::Value event;   
//    Json::Value vec1(Json::arrayValue);
//    vec1.append(Json::Value(1));
//    vec1.append(Json::Value(2));
//    vec1.append(Json::Value(3));
//
//    event["competitors"]["home"]["name"] = "Liverpool";
//    event["competitors"]["away"]["code"] = 89223;
//    event["competitors"]["away"]["name"] = "Aston Villa";
//    event["competitors"]["away"]["code"]=vec1;
//
//    std::cout << event << std::endl;
//////////////////////////////////////// write json ///////////////
//    std::ofstream file_id;
//    file_id.open("file.json");
//    Json::StyledWriter styledWriter;
//    file_id << styledWriter.write(event);
//
//    file_id.close();
//////////////////JSON WRITE CHECK//////////////////////////////////////////////////////
  FILE *f;
  char st1[max_size];
  char bestw[N][max_size];
  char file_name[max_size], st[100][max_size];
  float dist, len, bestd[N], vec[max_size];
  long long words, size, a, b, c, d, cn, bi[100];
  char ch;
  float *M;
  char *vocab;
  if (argc < 2) {
    printf("Usage: ./distance <FILE>\nwhere FILE contains word projections in the BINARY FORMAT\n");
    return 0;
  }
  strcpy(file_name, argv[1]);
  f = fopen(file_name, "rb");
  if (f == NULL) {
    printf("Input file not found\n");
    return -1;
  }
  fscanf(f, "%lld", &words);
  fscanf(f, "%lld", &size);
  vocab = (char *)malloc((long long)words * max_w * sizeof(char));
  M = (float *)malloc((long long)words * (long long)size * sizeof(float));
  if (M == NULL) {
    printf("Cannot allocate memory: %lld MB    %lld  %lld\n", (long long)words * size * sizeof(float) / 1048576, words, size);
    return -1;
  }
//////////////////////  READ VECTOR FILE /////////////////////////////
 Json::Value event;   
 std::ofstream outfile;
 //outfile.open("data.json", std::ios_base::app);
 outfile.open("vocab.txt");
  for (b = 0; b < words; b++) {
    Json::Value vec1(Json::arrayValue);
    fscanf(f, "%s%c", &vocab[b * max_w], &ch);
    for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);
    len = 0;
    for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
    len = sqrt(len);
    for (a = 0; a < size; a++) M[a + b * size] /= len;
    //cout<< M[a + b * size] <<"sdssssssssssssssssssssssss"<< endl;
   ///////////////// write vectors in json ///////////////
    for (a = 0; a < size; a++); {
     vec1.append(Json::Value(M[a + b * size]));
   ///////////////// write vectors in json ///////////////
     //vec1.append(Json::Value(2));
     //vec1.append(Json::Value(3));
    }
     outfile << vec1;
    //event[&vocab[b * max_w]] = vec1;
   ///////////////// write vectors in json ///////////////
  }
  fclose(f);
//////////////////////  READ VECTOR FILE /////////////////////////////

  while (1) {
    for (a = 0; a < N; a++) bestd[a] = 0;
    for (a = 0; a < N; a++) bestw[a][0] = 0;
    printf("Enter word or sentence (EXIT to break): ");
    a = 0;
    while (1) {
      st1[a] = fgetc(stdin);
      if ((st1[a] == '\n') || (a >= max_size - 1)) {
        st1[a] = 0;
        break;
      }
      a++;
    }
    if (!strcmp(st1, "EXIT")) break;
    cn = 0;
    b = 0;
    c = 0;
    while (1) {
      st[cn][b] = st1[c];
      b++;
      c++;
      st[cn][b] = 0;
      if (st1[c] == 0) break;
      if (st1[c] == ' ') {
        cn++;
        b = 0;
        c++;
      }
    }
    cn++;
    for (a = 0; a < cn; a++) {
      for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st[a])) break;
      if (b == words) b = -1;
      bi[a] = b;
      //printf("---%lli ---%lli----%lli---%s", words, a, b, vocab);
      //cout<<"djhbfvjs"<<vocab<<"\t"<<a<<"\t"<<b<<"\t"<<st[a]<<"\t"<<"\t"<<cn<<"\t"<<&vocab[b * max_w]<<"\t"<<&vocab[(b+2) * max_w]<<endl<<"TESTINGGGGGGGGGGG";
      printf("\nWord: %s  Position in vocabulary: %lld\n", st[a], bi[a]);
      if (b == -1) {
        printf("Out of dictionary word!\n");
        break;
      }
    }
    if (b == -1) continue;
    printf("\n                                              Word       Cosine distance\n------------------------------------------------------------------------\n");
    for (a = 0; a < size; a++) vec[a] = 0;
    for (b = 0; b < cn; b++) {
      if (bi[b] == -1) continue;
      for (a = 0; a < size; a++) vec[a] += M[a + bi[b] * size];
    }
    len = 0;
    for (a = 0; a < size; a++) len += vec[a] * vec[a];
    len = sqrt(len);
    for (a = 0; a < size; a++) vec[a] /= len;
    for (a = 0; a < N; a++) bestd[a] = 0;
    for (a = 0; a < N; a++) bestw[a][0] = 0;
    for (c = 0; c < words; c++) {
      a = 0;
      for (b = 0; b < cn; b++) if (bi[b] == c) a = 1;
      if (a == 1) continue;
      dist = 0;
      for (a = 0; a < size; a++) dist += vec[a] * M[a + c * size];
      //cout<<vec[a]<<"TESTTTTTTTTTTTT\t"<<M[a+c*size]<<endl<<"TESTINGGGGGGGGGGGGGG";
      //cout<<&vocab[c*max_w]<<"     shbdcjsbcsbcjdbcjsbcjsbcjsdbc"<<endl;////////////////////////////
      outfile<<&vocab[c*max_w]<<endl;
      for (a = 0; a < N; a++) {
        if (dist > bestd[a]) {
          for (d = N - 1; d > a; d--) {
            bestd[d] = bestd[d - 1];
            strcpy(bestw[d], bestw[d - 1]);
          }
          bestd[a] = dist;
          strcpy(bestw[a], &vocab[c * max_w]);
          break;
        }
      }
    }
    for (a = 0; a < N; a++) printf("%50s\t\t%f\n", bestw[a], bestd[a]);

    ///////////// To write vectors in file///////////////////////////
    //for (c = 0; c < words; c++){
    //  Json::Value vec1(Json::arrayValue);
    //  Json::Value event;
    //  for (a = 0; a < size; a++); {
    //    vec1.append(Json::Value(M[a + b * size]));
    //    cout<< M[a + b * size]<< endl;
    //   }
    //    event[&vocab[c * max_w]] = vec1;
    //    outfile << event;
    // if (c==2) break;
    // }
    ////////////////////////////////////////
  }
  return 0;
}
