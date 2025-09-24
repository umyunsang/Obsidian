# DAU_BigDataAnalytics_06_(Variable_Selection)

# 변수 선택(Variable Selection) 기법에 대한 코드

# Quick setup for pySpark

~~~python
!pip install pyspark
!apt install openjdk-8-jdk-headless -qq
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"

~~~

~~~python
# pyspark 모듈 및 sql 라이브러리 설치
import pyspark
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf

~~~

~~~python
# create the session
conf = SparkConf().set("spark.ui.port", "4050")

# create the context
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()

~~~

~~~python
spark

~~~

# 데이터셋 다운로드
본 데이터는 포트투갈 은행의 직접 마케팅 캠페인에 관련된 데이터임. 마케팅 캠페인은 전화에 기반하여 실시되었음. 
https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

# 목표: 고객이 정기예금에 가입하였는지 안했는지를 예측하는 것

~~~python
!wget -q https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip

~~~

~~~python
!mkdir data
!unzip bank.zip -d data/

~~~
Output:
~~~
Archive:  bank.zip
  inflating: data/bank-full.csv      
  inflating: data/bank-names.txt     
  inflating: data/bank.csv           

~~~

데이터 읽기 및
타겟 변수 (y) 설정: 고객이 정기예금에 가입했는지에 대한 여부 (y/n)

~~~python
filename = "data/bank-full.csv"
target_variable_name = "y"

~~~

~~~python
df = spark.read.csv(filename, header=True, inferSchema=True, sep=';')
df.show()
# df.schema 확인

~~~

스키마 출력

~~~python
df.printSchema()

~~~

education 컬럼의 값을 그룹화하여 갯수를 계산

~~~python
df.groupBy('education').count().show()

~~~

~~~python
df.groupBy(target_variable_name).count().show()

~~~

~~~python
df.groupBy(['education',target_variable_name]).count().show()

~~~

Questions 1. 다수개의 features와 target 값을 group화 한뒤 개수를 출력하시오

~~~python
from pyspark.sql.functions import * 
df.groupBy(target_variable_name).agg({'balance':'avg', 'age': 'avg'}).show()

~~~

~~~python
from pyspark.sql.functions import approxCountDistinct, countDistinct

def cardinality_calculation(df, cut_off=1):
    cardinality = df.select(*[approxCountDistinct(c).alias(c) for c in df.columns])
    
    ## convert to pandas for efficient calculations
    final_cardinality_df = cardinality.toPandas().transpose()
    final_cardinality_df.reset_index(inplace=True) 
    final_cardinality_df.rename(columns={0:'Cardinality'}, inplace=True) 
    
    #select variables with cardinality of 1
    vars_selected = final_cardinality_df['index'][final_cardinality_df['Cardinality'] <= cut_off] 
    
    return final_cardinality_df, vars_selected

cardinality_df, cardinality_vars_selected = cardinality_calculation(df)

~~~

~~~python
cardinality_df

~~~

~~~python
from pyspark.sql.functions import count, when, isnan, col

# miss_percentage 80%
def missing_calculation(df, miss_percentage=0.80):
    
    # checks for both NaN and null values
    missing = df.select(*[count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns])
    length_df = df.count()
    ## convert to pandas for efficient calculations
    final_missing_df = missing.toPandas().transpose()
    final_missing_df.reset_index(inplace=True) 
    final_missing_df.rename(columns={0:'missing_count'}, inplace=True) 
    final_missing_df['missing_percentage'] = final_missing_df['missing_count']/length_df
    
    #select variables with cardinality of 1
    vars_selected = final_missing_df['index'][final_missing_df['missing_percentage'] >= miss_percentage] 
    
    return final_missing_df, vars_selected

~~~

~~~python
df.describe().toPandas()

~~~
Output:
~~~
  summary                 age      job  ...            previous poutcome      y
0   count               45211    45211  ...               45211    45211  45211
1    mean   40.93621021432837     None  ...  0.5803233726305546     None   None
2  stddev  10.618762040975408     None  ...  2.3034410449312204     None   None
3     min                  18   admin.  ...                   0  failure     no
4     max                  95  unknown  ...                 275  unknown    yes

[5 rows x 18 columns]

~~~

~~~python
missing_df, missing_vars_selected = missing_calculation(df)

~~~

~~~python
missing_df

~~~
Output:
~~~
        index  missing_count  missing_percentage
0         age              0                 0.0
1         job              0                 0.0
2     marital              0                 0.0
3   education              0                 0.0
4     default              0                 0.0
5     balance              0                 0.0
6     housing              0                 0.0
7        loan              0                 0.0
8     contact              0                 0.0
9         day              0                 0.0
10      month              0                 0.0
11   duration              0                 0.0
12   campaign              0                 0.0
13      pdays              0                 0.0
14   previous              0                 0.0
15   poutcome              0                 0.0
16          y              0                 0.0

~~~

String 변수를 가진 컬럼을 식별

~~~python
def variable_type(df):
    
    vars_list = df.dtypes
    char_vars = []
    num_vars = []
    for i in vars_list:
        if i[1] in ('string'):
            char_vars.append(i[0])
        else:
            num_vars.append(i[0])
    
    return char_vars, num_vars

~~~

~~~python
char_vars, num_vars = variable_type(df)

~~~

~~~python
char_vars

~~~
Output:
~~~
['job',
 'marital',
 'education',
 'default',
 'housing',
 'loan',
 'contact',
 'month',
 'poutcome',
 'y']

~~~

~~~python
num_vars

~~~
Output:
~~~
['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

~~~

특정변환기를 선택된 컬럼에 적용

~~~python
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

def category_to_index(df, char_vars):
    char_df = df.select(char_vars)
    indexers = [StringIndexer(inputCol=c, outputCol=c+"_index", handleInvalid="keep") for c in char_df.columns]
    pipeline = Pipeline(stages=indexers)
    char_labels = pipeline.fit(char_df)
    df = char_labels.transform(df)
    return df, char_labels

~~~

~~~python
df, char_labels = category_to_index(df, char_vars)

~~~

~~~python
df = df.select([c for c in df.columns if c not in char_vars])

~~~

~~~python
def rename_columns(df, char_vars):
    mapping = dict(zip([i + '_index' for i in char_vars], char_vars))
    df = df.select([col(c).alias(mapping.get(c, c)) for c in df.columns])
    return df

~~~

~~~python
df = rename_columns(df, char_vars)

~~~

~~~python
df.dtypes

~~~
Output:
~~~
[('age', 'int'),
 ('balance', 'int'),
 ('day', 'int'),
 ('duration', 'int'),
 ('campaign', 'int'),
 ('pdays', 'int'),
 ('previous', 'int'),
 ('job', 'double'),
 ('marital', 'double'),
 ('education', 'double'),
 ('default', 'double'),
 ('housing', 'double'),
 ('loan', 'double'),
 ('contact', 'double'),
 ('month', 'double'),
 ('poutcome', 'double'),
 ('y', 'double')]

~~~

~~~python
df.describe().toPandas()

~~~
Output:
~~~
  summary                 age  ...             poutcome                    y
0   count               45211  ...                45211                45211
1    mean   40.93621021432837  ...  0.29006215301585897  0.11698480458295547
2  stddev  10.618762040975408  ...   0.6984693494366165    0.321405732615664
3     min                  18  ...                  0.0                  0.0
4     max                  95  ...                  3.0                  1.0

[5 rows x 18 columns]

~~~

특징을 Assemble

~~~python
from pyspark.ml.feature import VectorAssembler

#assemble individual columns to one column - 'features'
def assemble_vectors(df, features_list, target_variable_name):
    stages = []
    #assemble vectors
    assembler = VectorAssembler(inputCols=features_list, outputCol='features')
    stages = [assembler]
    #select all the columns + target + newly created 'features' column
    selectedCols = [target_variable_name, 'features'] + features_list
    #use pipeline to process sequentially
    pipeline = Pipeline(stages=stages)
    #assembler model
    assembleModel = pipeline.fit(df)
    #apply assembler model on data
    df = assembleModel.transform(df).select(selectedCols)

    return df

~~~

~~~python
#exclude target variable and select all other feature vectors
features_list = df.columns
#features_list = char_vars #this option is used only for ChiSqselector
features_list.remove(target_variable_name)

~~~

~~~python
features_list

~~~
Output:
~~~
['age',
 'balance',
 'day',
 'duration',
 'campaign',
 'pdays',
 'previous',
 'job',
 'marital',
 'education',
 'default',
 'housing',
 'loan',
 'contact',
 'month',
 'poutcome']

~~~

~~~python
# apply the function on our dataframe
df = assemble_vectors(df, features_list, target_variable_name)

~~~

~~~python
df.show()

~~~
Output:
~~~
+---+--------------------+---+-------+---+--------+--------+-----+--------+----+-------+---------+-------+-------+----+-------+-----+--------+
|  y|            features|age|balance|day|duration|campaign|pdays|previous| job|marital|education|default|housing|loan|contact|month|poutcome|
+---+--------------------+---+-------+---+--------+--------+-----+--------+----+-------+---------+-------+-------+----+-------+-----+--------+
|0.0|(16,[0,1,2,3,4,5,...| 58|   2143|  5|     261|       1|   -1|       0| 1.0|    0.0|      1.0|    0.0|    0.0| 0.0|    1.0|  0.0|     0.0|
|0.0|(16,[0,1,2,3,4,5,...| 44|     29|  5|     151|       1|   -1|       0| 2.0|    1.0|      0.0|    0.0|    0.0| 0.0|    1.0|  0.0|     0.0|
|0.0|(16,[0,1,2,3,4,5,...| 33|      2|  5|      76|       1|   -1|       0| 7.0|    0.0|      0.0|    0.0|    0.0| 1.0|    1.0|  0.0|     0.0|
|0.0|(16,[0,1,2,3,4,5,...| 47|   1506|  5|      92|       1|   -1|       0| 0.0|    0.0|      3.0|    0.0|    0.0| 0.0|    1.0|  0.0|     0.0|
|0.0|[33.0,1.0,5.0,198...| 33|      1|  5|     198|       1|   -1|       0|11.0|    1.0|      3.0|    0.0|    1.0| 0.0|    1.0|  0.0|     0.0|
|0.0|(16,[0,1,2,3,4,5,...| 35|    231|  5|     139|       1|   -1|       0| 1.0|    0.0|      1.0|    0.0|    0.0| 0.0|    1.0|  0.0|     0.0|
|0.0|[28.0,447.0,5.0,2...| 28|    447|  5|     217|       1|   -1|       0| 1.0|    1.0|      1.0|    0.0|    0.0| 1.0|    1.0|  0.0|     0.0|
|0.0|[42.0,2.0,5.0,380...| 42|      2|  5|     380|       1|   -1|       0| 7.0|    2.0|      1.0|    1.0|    0.0| 0.0|    1.0|  0.0|     0.0|
|0.0|(16,[0,1,2,3,4,5,...| 58|    121|  5|      50|       1|   -1|       0| 5.0|    0.0|      2.0|    0.0|    0.0| 0.0|    1.0|  0.0|     0.0|
|0.0|(16,[0,1,2,3,4,5,...| 43|    593|  5|      55|       1|   -1|       0| 2.0|    1.0|      0.0|    0.0|    0.0| 0.0|    1.0|  0.0|     0.0|
|0.0|(16,[0,1,2,3,4,5,...| 41|    270|  5|     222|       1|   -1|       0| 3.0|    2.0|      0.0|    0.0|    0.0| 0.0|    1.0|  0.0|     0.0|
|0.0|(16,[0,1,2,3,4,5,...| 29|    390|  5|     137|       1|   -1|       0| 3.0|    1.0|      0.0|    0.0|    0.0| 0.0|    1.0|  0.0|     0.0|
|0.0|(16,[0,1,2,3,4,5,...| 53|      6|  5|     517|       1|   -1|       0| 2.0|    0.0|      0.0|    0.0|    0.0| 0.0|    1.0|  0.0|     0.0|
|0.0|(16,[0,1,2,3,4,5,...| 58|     71|  5|      71|       1|   -1|       0| 2.0|    0.0|      3.0|    0.0|    0.0| 0.0|    1.0|  0.0|     0.0|
|0.0|(16,[0,1,2,3,4,5,...| 57|    162|  5|     174|       1|   -1|       0| 4.0|    0.0|      0.0|    0.0|    0.0| 0.0|    1.0|  0.0|     0.0|
|0.0|(16,[0,1,2,3,4,5,...| 51|    229|  5|     353|       1|   -1|       0| 5.0|    0.0|      2.0|    0.0|    0.0| 0.0|    1.0|  0.0|     0.0|
|0.0|[45.0,13.0,5.0,98...| 45|     13|  5|      98|       1|   -1|       0| 3.0|    1.0|      3.0|    0.0|    0.0| 0.0|    1.0|  0.0|     0.0|
|0.0|(16,[0,1,2,3,4,5,...| 57|     52|  5|      38|       1|   -1|       0| 0.0|    0.0|      2.0|    0.0|    0.0| 0.0|    1.0|  0.0|     0.0|
|0.0|(16,[0,1,2,3,4,5,...| 60|     60|  5|     219|       1|   -1|       0| 5.0|    0.0|      2.0|    0.0|    0.0| 0.0|    1.0|  0.0|     0.0|
|0.0|(16,[0,2,3,4,5,7,...| 33|      0|  5|      54|       1|   -1|       0| 4.0|    0.0|      0.0|    0.0|    0.0| 0.0|    1.0|  0.0|     0.0|
+---+--------------------+---+-------+---+--------+--------+-----+--------+----+-------+---------+-------+-------+----+-------+-----+--------+
only showing top 20 rows


~~~

~~~python
df.schema

~~~
Output:
~~~
StructType(List(StructField(y,DoubleType,false),StructField(features,VectorUDT,true),StructField(age,IntegerType,true),StructField(balance,IntegerType,true),StructField(day,IntegerType,true),StructField(duration,IntegerType,true),StructField(campaign,IntegerType,true),StructField(pdays,IntegerType,true),StructField(previous,IntegerType,true),StructField(job,DoubleType,false),StructField(marital,DoubleType,false),StructField(education,DoubleType,false),StructField(default,DoubleType,false),StructField(housing,DoubleType,false),StructField(loan,DoubleType,false),StructField(contact,DoubleType,false),StructField(month,DoubleType,false),StructField(poutcome,DoubleType,false)))

~~~

~~~python
df.schema["features"].metadata["ml_attr"]["attrs"]

~~~
Output:
~~~
{'nominal': [{'idx': 7,
   'name': 'job',
   'vals': ['blue-collar',
    'management',
    'technician',
    'admin.',
    'services',
    'retired',
    'self-employed',
    'entrepreneur',
    'unemployed',
    'housemaid',
    'student',
    'unknown',
    '__unknown']},
  {'idx': 8,
   'name': 'marital',
   'vals': ['married', 'single', 'divorced', '__unknown']},
  {'idx': 9,
   'name': 'education',
   'vals': ['secondary', 'tertiary', 'primary', 'unknown', '__unknown']},
  {'idx': 10, 'name': 'default', 'vals': ['no', 'yes', '__unknown']},
  {'idx': 11, 'name': 'housing', 'vals': ['yes', 'no', '__unknown']},
  {'idx': 12, 'name': 'loan', 'vals': ['no', 'yes', '__unknown']},
  {'idx': 13,
   'name': 'contact',
   'vals': ['cellular', 'unknown', 'telephone', '__unknown']},
  {'idx': 14,
   'name': 'month',
   'vals': ['may',
    'jul',
    'aug',
    'jun',
    'nov',
    'apr',
    'feb',
    'jan',
    'oct',
    'sep',
    'mar',
    'dec',
    '__unknown']},
  {'idx': 15,
   'name': 'poutcome',
   'vals': ['unknown', 'failure', 'other', 'success', '__unknown']}],
 'numeric': [{'idx': 0, 'name': 'age'},
  {'idx': 1, 'name': 'balance'},
  {'idx': 2, 'name': 'day'},
  {'idx': 3, 'name': 'duration'},
  {'idx': 4, 'name': 'campaign'},
  {'idx': 5, 'name': 'pdays'},
  {'idx': 6, 'name': 'previous'}]}

~~~

~~~python
import pandas as pd
features_df = None
for k, v in df.schema["features"].metadata["ml_attr"]["attrs"].items():
    if features_df is None:
      features_df = pd.DataFrame(v)
    else:
      features_df= pd.concat([features_df, pd.DataFrame(v)], axis=0)

~~~

~~~python
features_df = features_df.loc[:, ['idx', 'name']]
features_df

~~~
Output:
~~~
   idx       name
0    0        age
1    1    balance
2    2        day
3    3   duration
4    4   campaign
5    5      pdays
6    6   previous
0    7        job
1    8    marital
2    9  education
3   10    default
4   11    housing
5   12       loan
6   13    contact
7   14      month
8   15   poutcome

~~~

Scaled input vectors assembled

~~~python
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler

#assemble and scale individual columns to one column - 'features2'
def scaled_assemble_vectors(df, features_list, target_variable_name):
    stages = []
    #assemble vectors
    assembler = VectorAssembler(inputCols=features_list, outputCol='assembled_features')
    scaler = StandardScaler(inputCol=assembler.getOutputCol(), outputCol='features2')
    stages = [assembler, scaler]
    #select all the columns + target + newly created 'features' column
    selectedCols = [target_variable_name, 'features2'] + features_list
    #use pipeline to process sequentially
    pipeline = Pipeline(stages=stages)
    #assembler model
    scaleAssembleModel = pipeline.fit(df)
    #apply assembler model on data
    df = scaleAssembleModel.transform(df).select(selectedCols)
    return df

~~~

~~~python
features_list = df.columns
features_list.remove(target_variable_name)

~~~

~~~python
df = scaled_assemble_vectors(df, features_list, target_variable_name)

~~~

~~~python
df.show()

~~~
Output:
~~~
+---+--------------------+--------------------+---+-------+---+--------+--------+-----+--------+----+-------+---------+-------+-------+----+-------+-----+--------+
|  y|           features2|            features|age|balance|day|duration|campaign|pdays|previous| job|marital|education|default|housing|loan|contact|month|poutcome|
+---+--------------------+--------------------+---+-------+---+--------+--------+-----+--------+----+-------+---------+-------+-------+----+-------+-----+--------+
|0.0|(32,[0,1,2,3,4,5,...|(16,[0,1,2,3,4,5,...| 58|   2143|  5|     261|       1|   -1|       0| 1.0|    0.0|      1.0|    0.0|    0.0| 0.0|    1.0|  0.0|     0.0|
|0.0|(32,[0,1,2,3,4,5,...|(16,[0,1,2,3,4,5,...| 44|     29|  5|     151|       1|   -1|       0| 2.0|    1.0|      0.0|    0.0|    0.0| 0.0|    1.0|  0.0|     0.0|
|0.0|(32,[0,1,2,3,4,5,...|(16,[0,1,2,3,4,5,...| 33|      2|  5|      76|       1|   -1|       0| 7.0|    0.0|      0.0|    0.0|    0.0| 1.0|    1.0|  0.0|     0.0|
|0.0|(32,[0,1,2,3,4,5,...|(16,[0,1,2,3,4,5,...| 47|   1506|  5|      92|       1|   -1|       0| 0.0|    0.0|      3.0|    0.0|    0.0| 0.0|    1.0|  0.0|     0.0|
|0.0|[3.10770689395434...|[33.0,1.0,5.0,198...| 33|      1|  5|     198|       1|   -1|       0|11.0|    1.0|      3.0|    0.0|    1.0| 0.0|    1.0|  0.0|     0.0|
|0.0|(32,[0,1,2,3,4,5,...|(16,[0,1,2,3,4,5,...| 35|    231|  5|     139|       1|   -1|       0| 1.0|    0.0|      1.0|    0.0|    0.0| 0.0|    1.0|  0.0|     0.0|
|0.0|[2.63684221305217...|[28.0,447.0,5.0,2...| 28|    447|  5|     217|       1|   -1|       0| 1.0|    1.0|      1.0|    0.0|    0.0| 1.0|    1.0|  0.0|     0.0|
|0.0|[3.95526331957825...|[42.0,2.0,5.0,380...| 42|      2|  5|     380|       1|   -1|       0| 7.0|    2.0|      1.0|    1.0|    0.0| 0.0|    1.0|  0.0|     0.0|
|0.0|(32,[0,1,2,3,4,5,...|(16,[0,1,2,3,4,5,...| 58|    121|  5|      50|       1|   -1|       0| 5.0|    0.0|      2.0|    0.0|    0.0| 0.0|    1.0|  0.0|     0.0|
|0.0|(32,[0,1,2,3,4,5,...|(16,[0,1,2,3,4,5,...| 43|    593|  5|      55|       1|   -1|       0| 2.0|    1.0|      0.0|    0.0|    0.0| 0.0|    1.0|  0.0|     0.0|
|0.0|(32,[0,1,2,3,4,5,...|(16,[0,1,2,3,4,5,...| 41|    270|  5|     222|       1|   -1|       0| 3.0|    2.0|      0.0|    0.0|    0.0| 0.0|    1.0|  0.0|     0.0|
|0.0|(32,[0,1,2,3,4,5,...|(16,[0,1,2,3,4,5,...| 29|    390|  5|     137|       1|   -1|       0| 3.0|    1.0|      0.0|    0.0|    0.0| 0.0|    1.0|  0.0|     0.0|
|0.0|(32,[0,1,2,3,4,5,...|(16,[0,1,2,3,4,5,...| 53|      6|  5|     517|       1|   -1|       0| 2.0|    0.0|      0.0|    0.0|    0.0| 0.0|    1.0|  0.0|     0.0|
|0.0|(32,[0,1,2,3,4,5,...|(16,[0,1,2,3,4,5,...| 58|     71|  5|      71|       1|   -1|       0| 2.0|    0.0|      3.0|    0.0|    0.0| 0.0|    1.0|  0.0|     0.0|
|0.0|(32,[0,1,2,3,4,5,...|(16,[0,1,2,3,4,5,...| 57|    162|  5|     174|       1|   -1|       0| 4.0|    0.0|      0.0|    0.0|    0.0| 0.0|    1.0|  0.0|     0.0|
|0.0|(32,[0,1,2,3,4,5,...|(16,[0,1,2,3,4,5,...| 51|    229|  5|     353|       1|   -1|       0| 5.0|    0.0|      2.0|    0.0|    0.0| 0.0|    1.0|  0.0|     0.0|
|0.0|(32,[0,1,2,3,4,5,...|[45.0,13.0,5.0,98...| 45|     13|  5|      98|       1|   -1|       0| 3.0|    1.0|      3.0|    0.0|    0.0| 0.0|    1.0|  0.0|     0.0|
|0.0|(32,[0,1,2,3,4,5,...|(16,[0,1,2,3,4,5,...| 57|     52|  5|      38|       1|   -1|       0| 0.0|    0.0|      2.0|    0.0|    0.0| 0.0|    1.0|  0.0|     0.0|
|0.0|(32,[0,1,2,3,4,5,...|(16,[0,1,2,3,4,5,...| 60|     60|  5|     219|       1|   -1|       0| 5.0|    0.0|      2.0|    0.0|    0.0| 0.0|    1.0|  0.0|     0.0|
|0.0|(32,[0,2,3,4,5,7,...|(16,[0,2,3,4,5,7,...| 33|      0|  5|      54|       1|   -1|       0| 4.0|    0.0|      0.0|    0.0|    0.0| 0.0|    1.0|  0.0|     0.0|
+---+--------------------+--------------------+---+-------+---+--------+--------+-----+--------+----+-------+---------+-------+-------+----+-------+-----+--------+
only showing top 20 rows


~~~

Built-in Variable Selection Process: Without Target

Principal Component Analysis

~~~python
df.describe().toPandas()

~~~
Output:
~~~
  summary                    y  ...               month             poutcome
0   count                45211  ...               45211                45211
1    mean  0.11698480458295547  ...  2.4309570679701844  0.29006215301585897
2  stddev    0.321405732615664  ...   2.494990479670114   0.6984693494366165
3     min                  0.0  ...                 0.0                  0.0
4     max                  1.0  ...                11.0                  3.0

[5 rows x 18 columns]

~~~

~~~python
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors

pca = PCA(k=3, inputCol="features", outputCol="pcaFeatures")
model = pca.fit(df)

result = model.transform(df).select("pcaFeatures")
result.show(truncate=False)

~~~
Output:
~~~
+------------------------------------------------------------+
|pcaFeatures                                                 |
+------------------------------------------------------------+
|[-2143.495364773581,-257.0420740676505,1.2449712753048212]  |
|[-29.29226175164628,-150.92888640669165,1.0493026899279903] |
|[-2.150889773941882,-75.98191250436612,1.0760651955381237]  |
|[-1506.1823305908117,-89.21547154672699,1.2683999536976271] |
|[-1.3750908349448236,-197.9835617549482,0.9892427009816414] |
|[-231.26679712386618,-138.56034919503713,1.0576960891396285]|
|[-447.4072782675641,-216.16541383839743,1.0066666548031231] |
|[-2.712326444744223,-379.97858281643545,0.8751504698368189] |
|[-121.11144848214049,-49.75636025970778,1.172585359192404]  |
|[-593.1146061641075,-53.89364261832033,1.1817022732262128]  |
|[-270.4212341323417,-221.48699661827914,1.0170824073205755] |
|[-390.26081676654087,-136.27016909339133,1.0625336862699843]|
|[-6.967616100129188,-516.9674558248696,0.8003545768071488]  |
|[-71.15006476185233,-70.84798774752198,1.1499619332808761]  |
|[-162.33874829001974,-173.68095250690791,1.0823345183248174]|
|[-229.66539702564629,-352.5590565974901,0.9435034899633544] |
|[-13.195336090120463,-97.9581534500072,1.090278500900193]   |
|[-52.08909854893981,-37.88319180402837,1.1682296517559208]  |
|[-60.422642726801485,-218.86732123338794,1.0462594933058524]|
|[-0.11044203745873847,-53.98554342962398,1.0903138050958234]|
+------------------------------------------------------------+
only showing top 20 rows


~~~

~~~python
model.pc.toArray()

~~~
Output:
~~~
array([[-3.41021399e-04,  2.79524640e-04,  2.58353293e-03],
       [-9.99998245e-01,  1.83654726e-03,  1.13892524e-04],
       [-1.22934480e-05,  9.79995613e-04,  7.79347982e-03],
       [-1.83671689e-03, -9.99996986e-01, -7.36955549e-04],
       [ 1.48468991e-05,  1.01391994e-03,  2.75121381e-03],
       [-1.13085547e-04,  7.49207153e-04, -9.99889046e-01],
       [-1.26153895e-05, -6.36100089e-06, -1.04654388e-02],
       [-1.78789640e-05, -4.11817349e-05,  5.51411389e-04],
       [ 6.41085932e-06, -5.23364803e-05, -1.45349520e-04],
       [-1.11185424e-05,  1.30366514e-05,  2.01500982e-04],
       [ 2.91665702e-06,  4.42643869e-06,  3.95562163e-05],
       [-1.12221341e-05,  1.26153926e-05,  6.17569266e-04],
       [ 1.01623400e-05,  1.50687571e-05,  8.23933054e-05],
       [-5.68377754e-07,  6.95393403e-05,  1.03951369e-03],
       [-7.60886236e-05, -1.16754927e-04, -3.24662847e-03],
       [-8.55162111e-06, -6.01853226e-05, -4.94522998e-03]])

~~~

~~~python
model.explainedVariance

~~~
Output:
~~~
DenseVector([0.9918, 0.0071, 0.0011])

~~~

~~~python
import matplotlib.pyplot as plt
import numpy as np
x = []
for i in range(0, len(model.explainedVariance)):
    x.append('PC' + str(i + 1))
y = np.array(model.explainedVariance)
z = np.cumsum(model.explainedVariance)
plt.xlabel('Principal Components')
plt.ylabel('Variance Explained')
plt.bar(x, y)
plt.plot(x, z)

~~~
Output:
~~~
[<matplotlib.lines.Line2D at 0x7f617e6be590>]<Figure size 432x288 with 1 Axes>

~~~

~~~python
pca = PCA(k=3, inputCol="features2", outputCol="pcaFeatures")
model = pca.fit(df)

result = model.transform(df).select("pcaFeatures")
result.show(truncate=False)

~~~
Output:
~~~
+------------------------------------------------------------+
|pcaFeatures                                                 |
+------------------------------------------------------------+
|[-0.6567808835228072,4.3121948711553015,-4.3937648469809]   |
|[-0.5905533920179786,2.7430023739115277,-2.0129736282025625]|
|[-0.7923309293106022,2.428517822202711,-1.412875839724541]  |
|[-0.7779852947451491,4.298302632066033,-5.322505502928547]  |
|[-0.5280039534401401,6.559141503669079,-1.4081500208397828] |
|[-0.7028277580737647,2.7093407342344897,-3.226355647066135] |
|[-0.8104872327792136,1.429021001344571,-1.619935441334062]  |
|[-0.9758735335487279,2.862044859015957,0.4335559756415678]  |
|[-0.7126916038562845,5.175637716397523,-4.531453571469944]  |
|[-0.5939132026124012,2.7467852504302592,-2.0709944257197654]|
|[-0.48367202786573626,2.584502296306621,-0.8133472106594244]|
|[-0.5574665926506135,2.079820030407318,-1.1695266299980636] |
|[-0.6052039081600511,3.4943812853193674,-3.06065582479699]  |
|[-0.7917889584107352,5.104448480115379,-5.533708763513019]  |
|[-0.634131428409618,4.061248668356405,-3.2668691358888577]  |
|[-0.6515302947285277,4.807107975840059,-3.9602262772305346] |
|[-0.6913724174639284,4.317335145410398,-3.8524349637084288] |
|[-0.7928529953941135,4.2455389078940895,-5.143335647201441] |
|[-0.6860141754953244,5.301081379539062,-4.4986208881179595] |
|[-0.6396798564710514,2.621609196420631,-2.1560847881880587] |
+------------------------------------------------------------+
only showing top 20 rows


~~~

~~~python
model.explainedVariance

~~~
Output:
~~~
DenseVector([0.1434, 0.0987, 0.0787])

~~~

Singluar Value Decomposition

~~~python
df_svd_vector = df.rdd.map(lambda x: x['features'].toArray())

~~~

~~~python
df_svd_vector

~~~
Output:
~~~
PythonRDD[208] at RDD at PythonRDD.scala:53

~~~

~~~python
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import RowMatrix

mat = RowMatrix(df_svd_vector)

# Compute the top 5 singular values and corresponding singular vectors.
svd = mat.computeSVD(5, computeU=True)
U = svd.U       # The U factor is a RowMatrix.
s = svd.s       # The singular values are stored in a local dense vector.
V = svd.V       # The V factor is a local dense matrix.

~~~

~~~python
U.rows.collect()

~~~
Output:
~~~
[DenseVector([-0.003, 0.0026, 0.0011, 0.0048, 0.0087]),
 DenseVector([-0.0, 0.0021, 0.0005, 0.0049, 0.006]),
 DenseVector([-0.0, 0.001, 0.0002, 0.0042, 0.0039]),
 DenseVector([-0.0021, 0.0006, 0.0004, 0.0053, 0.0066]),
 DenseVector([-0.0, 0.0027, 0.0007, 0.0029, 0.0039]),
 DenseVector([-0.0003, 0.0018, 0.0005, 0.0037, 0.0043]),
 DenseVector([-0.0006, 0.0027, 0.0008, 0.0017, 0.0029]),
 DenseVector([-0.0, 0.0051, 0.0013, 0.0022, 0.0057]),
 DenseVector([-0.0002, 0.0007, 0.0001, 0.008, 0.0088]),
 DenseVector([-0.0008, 0.0005, 0.0002, 0.0056, 0.0058]),
 DenseVector([-0.0004, 0.0029, 0.0008, 0.0036, 0.0055]),
 DenseVector([-0.0006, 0.0017, 0.0005, 0.0028, 0.0031]),
 DenseVector([-0.0, 0.007, 0.0018, 0.0022, 0.0079]),
 DenseVector([-0.0001, 0.001, 0.0001, 0.0078, 0.0088]),
 DenseVector([-0.0002, 0.0023, 0.0005, 0.0065, 0.0086]),
 DenseVector([-0.0003, 0.0047, 0.0012, 0.0036, 0.0075]),
 DenseVector([-0.0, 0.0013, 0.0003, 0.0057, 0.0062]),
 DenseVector([-0.0001, 0.0005, 0.0, 0.008, 0.0086]),
 DenseVector([-0.0001, 0.003, 0.0007, 0.0065, 0.0092]),
 DenseVector([-0.0, 0.0008, 0.0001, 0.0045, 0.0039]),
 DenseVector([-0.001, 0.0032, 0.001, 0.0011, 0.0029]),
 DenseVector([-0.0011, 0.0019, 0.0006, 0.0062, 0.0084]),
 DenseVector([-0.0, 0.0022, 0.0005, 0.0031, 0.0037]),
 DenseVector([-0.0001, 0.0046, 0.0013, 0.0001, 0.0024]),
 DenseVector([-0.0, 0.0025, 0.0006, 0.0041, 0.0053]),
 DenseVector([0.0005, 0.0025, 0.0005, 0.0049, 0.0061]),
 DenseVector([-0.0004, 0.0039, 0.0011, 0.0025, 0.0051]),
 DenseVector([-0.0002, 0.0017, 0.0004, 0.0063, 0.0076]),
 DenseVector([0.0003, 0.0036, 0.0008, 0.0042, 0.0064]),
 DenseVector([-0.0004, 0.0046, 0.0013, 0.0015, 0.0045]),
 DenseVector([-0.0012, 0.0027, 0.0008, 0.0056, 0.0086]),
 DenseVector([-0.0005, 0.003, 0.0008, 0.0046, 0.007]),
 DenseVector([-0.0001, 0.0028, 0.0006, 0.0066, 0.0092]),
 DenseVector([-0.0, 0.0031, 0.0007, 0.0062, 0.009]),
 DenseVector([-0.015, -0.0003, 0.0024, -0.0009, 0.0071]),
 DenseVector([-0.0001, 0.0033, 0.0008, 0.0058, 0.0086]),
 DenseVector([-0.0, 0.0049, 0.0013, -0.0002, 0.0024]),
 DenseVector([-0.0001, 0.0223, 0.0061, -0.0105, 0.0082]),
 DenseVector([-0.0007, 0.0075, 0.0022, -0.0011, 0.0046]),
 DenseVector([-0.0, 0.0019, 0.0004, 0.0041, 0.0047]),
 DenseVector([-0.0036, 0.001, 0.0008, 0.0037, 0.006]),
 DenseVector([-0.0001, 0.0024, 0.0006, 0.0055, 0.0072]),
 DenseVector([-0.0001, 0.0003, -0.0, 0.0086, 0.0091]),
 DenseVector([-0.0008, 0.0198, 0.0055, -0.0087, 0.0083]),
 DenseVector([-0.0002, 0.0083, 0.0022, 0.0018, 0.0089]),
 DenseVector([0.0002, 0.0034, 0.0008, 0.0029, 0.0045]),
 DenseVector([0.0005, 0.005, 0.0011, 0.0049, 0.0088]),
 DenseVector([-0.0, 0.0031, 0.0007, 0.0041, 0.006]),
 DenseVector([-0.0, 0.0022, 0.0005, 0.0064, 0.0082]),
 DenseVector([-0.0, 0.0049, 0.0013, 0.0004, 0.0032]),
 DenseVector([-0.0018, 0.003, 0.001, 0.0045, 0.008]),
 DenseVector([0.0003, 0.0035, 0.0008, 0.0045, 0.0069]),
 DenseVector([-0.0, 0.0024, 0.0006, 0.0029, 0.0037]),
 DenseVector([0.0001, 0.0106, 0.0029, -0.0023, 0.0058]),
 DenseVector([0.0001, 0.002, 0.0005, 0.0022, 0.0022]),
 DenseVector([-0.0004, 0.0023, 0.0006, 0.0038, 0.0049]),
 DenseVector([-0.0006, 0.0012, 0.0004, 0.0044, 0.0049]),
 DenseVector([-0.0004, 0.0001, -0.0, 0.0068, 0.0066]),
 DenseVector([-0.0, 0.0025, 0.0006, 0.004, 0.0053]),
 DenseVector([-0.0003, 0.0237, 0.0066, -0.0128, 0.0068]),
 DenseVector([-0.0, 0.0019, 0.0005, 0.0034, 0.0037]),
 DenseVector([-0.0014, 0.0105, 0.003, -0.0015, 0.0079]),
 DenseVector([-0.0004, 0.0021, 0.0005, 0.0065, 0.0086]),
 DenseVector([-0.0011, 0.0049, 0.0015, 0.0004, 0.0039]),
 DenseVector([-0.0002, 0.0048, 0.0012, 0.0033, 0.0071]),
 DenseVector([-0.0092, -0.0017, 0.001, 0.0037, 0.0072]),
 DenseVector([-0.0002, 0.0071, 0.0018, 0.003, 0.0093]),
 DenseVector([-0.0001, 0.0037, 0.0009, 0.0057, 0.009]),
 DenseVector([-0.0017, 0.0016, 0.0006, 0.0059, 0.0081]),
 DenseVector([-0.0172, -0.0032, 0.0021, -0.0021, 0.0039]),
 DenseVector([-0.0084, 0.0008, 0.0015, 0.0029, 0.0084]),
 DenseVector([-0.0, 0.0023, 0.0006, 0.0029, 0.0035]),
 DenseVector([-0.0004, 0.002, 0.0005, 0.0062, 0.008]),
 DenseVector([-0.0, 0.0039, 0.001, 0.0049, 0.0082]),
 DenseVector([-0.0027, 0.0016, 0.0008, 0.0036, 0.0058]),
 DenseVector([-0.0005, 0.0022, 0.0006, 0.0058, 0.0078]),
 DenseVector([-0.0008, 0.0026, 0.0008, 0.004, 0.006]),
 DenseVector([-0.0001, 0.0047, 0.0012, 0.0043, 0.0082]),
 DenseVector([-0.0, 0.0037, 0.0009, 0.0043, 0.007]),
 DenseVector([-0.0, 0.0028, 0.0007, 0.0059, 0.0082]),
 DenseVector([-0.0003, 0.0025, 0.0007, 0.0045, 0.0062]),
 DenseVector([-0.0002, 0.0028, 0.0007, 0.0046, 0.0066]),
 DenseVector([-0.001, -0.0, 0.0001, 0.0058, 0.0056]),
 DenseVector([-0.0033, 0.0129, 0.004, -0.0038, 0.0091]),
 DenseVector([-0.0002, 0.0033, 0.0008, 0.0042, 0.0065]),
 DenseVector([-0.0003, 0.007, 0.0019, 0.0017, 0.0075]),
 DenseVector([-0.0001, 0.0197, 0.0054, -0.0079, 0.0087]),
 DenseVector([-0.0019, 0.018, 0.0053, -0.0098, 0.0057]),
 DenseVector([-0.0, 0.0026, 0.0006, 0.0048, 0.0064]),
 DenseVector([-0.0007, 0.0022, 0.0006, 0.0063, 0.0085]),
 DenseVector([-0.0001, 0.0007, 0.0001, 0.0058, 0.0056]),
 DenseVector([-0.0002, 0.0028, 0.0008, 0.0022, 0.0033]),
 DenseVector([-0.0004, 0.0077, 0.0021, 0.0023, 0.0093]),
 DenseVector([-0.0001, 0.003, 0.0007, 0.0064, 0.0092]),
 DenseVector([0.0, 0.0024, 0.0005, 0.0066, 0.0086]),
 DenseVector([-0.0002, 0.0057, 0.0015, 0.0007, 0.0046]),
 DenseVector([-0.0006, 0.0037, 0.001, 0.0048, 0.0082]),
 DenseVector([-0.0001, 0.0014, 0.0002, 0.0078, 0.0092]),
 DenseVector([-0.0, 0.0028, 0.0007, 0.0036, 0.0051]),
 DenseVector([-0.0003, 0.0026, 0.0007, 0.0046, 0.0064]),
 DenseVector([0.0009, 0.0038, 0.0008, 0.0041, 0.0061]),
 DenseVector([-0.0001, 0.0017, 0.0004, 0.0065, 0.0078]),
 DenseVector([-0.0002, 0.0031, 0.0008, 0.0052, 0.0076]),
 DenseVector([-0.0003, 0.0007, 0.0001, 0.0081, 0.0089]),
 DenseVector([-0.0, 0.0054, 0.0015, -0.0002, 0.0028]),
 DenseVector([-0.0001, 0.0027, 0.0007, 0.0044, 0.0061]),
 DenseVector([-0.0002, 0.0025, 0.0006, 0.0049, 0.0066]),
 DenseVector([-0.0001, 0.0003, 0.0, 0.0049, 0.0041]),
 DenseVector([-0.0001, 0.0069, 0.0018, 0.0031, 0.0091]),
 DenseVector([-0.0008, 0.0112, 0.0031, -0.0029, 0.0064]),
 DenseVector([-0.0001, 0.0026, 0.0007, 0.0023, 0.0031]),
 DenseVector([-0.0001, 0.0019, 0.0004, 0.0053, 0.0064]),
 DenseVector([0.0, 0.0029, 0.0007, 0.006, 0.0084]),
 DenseVector([-0.0006, 0.0037, 0.0011, 0.0021, 0.0045]),
 DenseVector([-0.0001, 0.0015, 0.0003, 0.0076, 0.009]),
 DenseVector([-0.0002, 0.0033, 0.0008, 0.0038, 0.006]),
 DenseVector([-0.0005, 0.0068, 0.0019, 0.0003, 0.0055]),
 DenseVector([0.0001, 0.0049, 0.0013, 0.001, 0.004]),
 DenseVector([-0.0004, 0.0023, 0.0006, 0.0067, 0.009]),
 DenseVector([-0.0, 0.0014, 0.0002, 0.0074, 0.0086]),
 DenseVector([-0.0001, 0.0059, 0.0015, 0.0035, 0.0085]),
 DenseVector([-0.0, 0.0011, 0.0002, 0.0067, 0.0074]),
 DenseVector([-0.0, 0.0016, 0.0004, 0.0039, 0.004]),
 DenseVector([-0.0001, 0.0017, 0.0004, 0.0051, 0.0058]),
 DenseVector([-0.0, 0.0024, 0.0005, 0.0058, 0.0076]),
 DenseVector([-0.0002, 0.0035, 0.0009, 0.0021, 0.0039]),
 DenseVector([-0.0012, 0.0005, 0.0003, 0.0034, 0.0031]),
 DenseVector([0.0, 0.0011, 0.0002, 0.0043, 0.0041]),
 DenseVector([-0.0, 0.0019, 0.0005, 0.0032, 0.0035]),
 DenseVector([-0.0035, 0.0067, 0.0023, 0.0007, 0.0082]),
 DenseVector([-0.0017, 0.0086, 0.0026, 0.0002, 0.0083]),
 DenseVector([-0.0003, 0.0045, 0.0013, 0.001, 0.0037]),
 DenseVector([-0.0024, 0.0017, 0.0008, 0.003, 0.0048]),
 DenseVector([-0.0011, 0.001, 0.0003, 0.0067, 0.0081]),
 DenseVector([-0.0, 0.0017, 0.0004, 0.003, 0.0029]),
 DenseVector([-0.0001, 0.0026, 0.0007, 0.0014, 0.002]),
 DenseVector([-0.0, 0.0019, 0.0005, 0.0034, 0.0037]),
 DenseVector([-0.0, 0.001, 0.0002, 0.0057, 0.0058]),
 DenseVector([-0.0008, 0.0069, 0.002, -0.0012, 0.0038]),
 DenseVector([-0.0004, 0.0072, 0.002, 0.0008, 0.0065]),
 DenseVector([-0.0013, 0.0018, 0.0006, 0.0057, 0.0078]),
 DenseVector([-0.0, 0.0041, 0.0011, 0.0019, 0.0041]),
 DenseVector([-0.0001, 0.0007, 0.0, 0.008, 0.0086]),
 DenseVector([-0.0001, 0.0027, 0.0007, 0.0033, 0.0047]),
 DenseVector([-0.0001, 0.0013, 0.0002, 0.0076, 0.0089]),
 DenseVector([-0.0, 0.001, 0.0002, 0.0043, 0.0039]),
 DenseVector([-0.0, 0.0022, 0.0005, 0.0066, 0.0083]),
 DenseVector([-0.0004, 0.0032, 0.0008, 0.0044, 0.0068]),
 DenseVector([-0.0, 0.0025, 0.0006, 0.0044, 0.0058]),
 DenseVector([-0.0005, 0.0076, 0.0021, 0.0015, 0.008]),
 DenseVector([0.0, 0.0022, 0.0005, 0.0058, 0.0074]),
 DenseVector([-0.0, 0.0082, 0.0022, -0.0026, 0.0026]),
 DenseVector([-0.0001, 0.0272, 0.0075, -0.0165, 0.0057]),
 DenseVector([-0.0, 0.0012, 0.0002, 0.0049, 0.0051]),
 DenseVector([-0.0002, 0.0015, 0.0003, 0.0062, 0.0072]),
 DenseVector([-0.0, 0.0016, 0.0003, 0.0049, 0.0054]),
 DenseVector([-0.0027, -0.0001, 0.0003, 0.0061, 0.0073]),
 DenseVector([-0.0003, 0.0031, 0.0008, 0.0062, 0.0092]),
 DenseVector([-0.0011, 0.0009, 0.0003, 0.0064, 0.0075]),
 DenseVector([-0.0013, 0.0013, 0.0005, 0.0054, 0.0067]),
 DenseVector([-0.0024, 0.0007, 0.0005, 0.0053, 0.0068]),
 DenseVector([-0.0001, 0.0024, 0.0006, 0.0039, 0.0051]),
 DenseVector([-0.0003, 0.004, 0.0011, 0.0036, 0.0066]),
 DenseVector([-0.0018, 0.0069, 0.0022, -0.0007, 0.0052]),
 DenseVector([-0.0001, 0.0036, 0.0009, 0.0037, 0.0063]),
 DenseVector([-0.0001, 0.003, 0.0008, 0.0015, 0.0026]),
 DenseVector([-0.0004, 0.0013, 0.0003, 0.0065, 0.0076]),
 DenseVector([-0.002, 0.0026, 0.0009, 0.0048, 0.008]),
 DenseVector([-0.0003, 0.009, 0.0024, 0.0005, 0.0081]),
 DenseVector([-0.0, 0.0032, 0.0008, 0.0049, 0.0072]),
 DenseVector([-0.0001, 0.0142, 0.0039, -0.0064, 0.0045]),
 DenseVector([-0.0019, 0.0028, 0.001, 0.0033, 0.006]),
 DenseVector([-0.0, 0.0034, 0.0008, 0.0052, 0.0078]),
 DenseVector([-0.0006, 0.0017, 0.0005, 0.0036, 0.0043]),
 DenseVector([-0.0001, 0.0018, 0.0004, 0.0074, 0.0091]),
 DenseVector([-0.0016, 0.005, 0.0016, 0.0028, 0.0078]),
 DenseVector([-0.0001, 0.0024, 0.0006, 0.0052, 0.0068]),
 DenseVector([-0.0001, 0.0003, 0.0, 0.005, 0.004]),
 DenseVector([-0.0003, 0.0061, 0.0016, 0.0029, 0.008]),
 DenseVector([-0.0017, 0.0091, 0.0027, -0.0009, 0.0075]),
 DenseVector([-0.0001, 0.0042, 0.0011, 0.0013, 0.0036]),
 DenseVector([-0.0011, 0.0088, 0.0026, -0.0026, 0.0044]),
 DenseVector([-0.0001, 0.0144, 0.004, -0.0066, 0.0045]),
 DenseVector([-0.0004, 0.0055, 0.0015, 0.001, 0.0049]),
 DenseVector([-0.0003, 0.0019, 0.0005, 0.0038, 0.0044]),
 DenseVector([-0.0003, 0.0022, 0.0005, 0.0067, 0.0088]),
 DenseVector([-0.0062, 0.0022, 0.0016, 0.0006, 0.0052]),
 DenseVector([-0.0058, 0.0001, 0.0009, 0.0047, 0.0079]),
 DenseVector([-0.0001, 0.0046, 0.0012, 0.0013, 0.0041]),
 DenseVector([-0.0, 0.0075, 0.0021, -0.0015, 0.0036]),
 DenseVector([-0.003, 0.0014, 0.0008, 0.0048, 0.0073]),
 DenseVector([-0.0005, 0.0028, 0.0008, 0.0025, 0.0039]),
 DenseVector([-0.0001, 0.0019, 0.0004, 0.0066, 0.0081]),
 DenseVector([-0.0006, 0.0007, 0.0002, 0.0054, 0.0056]),
 DenseVector([-0.0004, 0.0038, 0.0011, 0.0018, 0.0041]),
 DenseVector([-0.0004, 0.004, 0.0011, 0.0015, 0.0039]),
 DenseVector([-0.0002, 0.0033, 0.0009, 0.003, 0.0049]),
 DenseVector([-0.0002, 0.0013, 0.0003, 0.0063, 0.0072]),
 DenseVector([-0.0006, 0.0033, 0.0009, 0.0034, 0.0059]),
 DenseVector([-0.002, 0.0005, 0.0003, 0.0075, 0.0092]),
 DenseVector([-0.0003, 0.0077, 0.0021, 0.0006, 0.0067]),
 DenseVector([-0.0, 0.0039, 0.001, 0.0041, 0.0069]),
 DenseVector([-0.0008, 0.0062, 0.0018, 0.0011, 0.0061]),
 DenseVector([-0.0004, 0.0081, 0.0023, -0.0015, 0.0044]),
 DenseVector([-0.0, 0.0063, 0.0017, 0.0001, 0.0044]),
 DenseVector([-0.008, 0.0025, 0.002, 0.0007, 0.007]),
 DenseVector([-0.0003, 0.0005, 0.0001, 0.0056, 0.0054]),
 DenseVector([-0.0007, 0.0032, 0.0009, 0.0032, 0.0055]),
 DenseVector([-0.0004, 0.0054, 0.0015, 0.0013, 0.0051]),
 DenseVector([-0.0001, 0.0029, 0.0007, 0.0039, 0.0057]),
 DenseVector([0.0005, 0.0041, 0.001, 0.0027, 0.0049]),
 DenseVector([-0.0005, 0.0028, 0.0007, 0.0054, 0.0078]),
 DenseVector([-0.0003, 0.0049, 0.0013, 0.0047, 0.009]),
 DenseVector([-0.0004, 0.0027, 0.0008, 0.003, 0.0045]),
 DenseVector([0.0, 0.0039, 0.001, 0.0049, 0.008]),
 DenseVector([-0.0007, 0.0021, 0.0006, 0.004, 0.0053]),
 DenseVector([-0.0001, 0.0046, 0.0012, 0.0034, 0.0067]),
 DenseVector([-0.0, 0.0055, 0.0014, 0.0034, 0.0078]),
 DenseVector([-0.0014, 0.002, 0.0007, 0.0046, 0.0064]),
 DenseVector([-0.0005, 0.0016, 0.0004, 0.0043, 0.005]),
 DenseVector([-0.0003, 0.0047, 0.0013, 0.002, 0.0053]),
 DenseVector([-0.0005, 0.0022, 0.0007, 0.0022, 0.0029]),
 DenseVector([0.0001, 0.0041, 0.001, 0.0021, 0.0043]),
 DenseVector([-0.0, 0.0019, 0.0004, 0.0067, 0.0082]),
 DenseVector([-0.0004, 0.0189, 0.0052, -0.0093, 0.0062]),
 DenseVector([-0.0004, 0.0016, 0.0004, 0.0057, 0.0068]),
 DenseVector([-0.0015, 0.0024, 0.0008, 0.0058, 0.0087]),
 DenseVector([-0.0, 0.0004, 0.0, 0.0047, 0.0039]),
 DenseVector([-0.0015, 0.0027, 0.0009, 0.0024, 0.0044]),
 DenseVector([-0.0021, 0.001, 0.0005, 0.0059, 0.0077]),
 DenseVector([0.0, 0.0003, -0.0, 0.0068, 0.0066]),
 DenseVector([-0.0006, 0.0096, 0.0027, -0.0035, 0.0036]),
 DenseVector([-0.0004, 0.0099, 0.0027, 0.0001, 0.0087]),
 DenseVector([-0.0026, 0.0008, 0.0006, 0.0057, 0.0077]),
 DenseVector([-0.0009, 0.0031, 0.0009, 0.0052, 0.0082]),
 DenseVector([-0.0, 0.0006, 0.0001, 0.0063, 0.0062]),
 DenseVector([-0.0005, 0.0023, 0.0006, 0.0058, 0.0078]),
 DenseVector([-0.0, 0.0011, 0.0002, 0.0047, 0.0047]),
 DenseVector([-0.0001, 0.0028, 0.0007, 0.0043, 0.0061]),
 DenseVector([-0.0, 0.0053, 0.0013, 0.003, 0.0071]),
 DenseVector([-0.0005, 0.0016, 0.0004, 0.0053, 0.0064]),
 DenseVector([-0.0002, 0.0094, 0.0026, -0.0013, 0.006]),
 DenseVector([-0.0, 0.0021, 0.0005, 0.0041, 0.0049]),
 DenseVector([-0.0006, 0.0014, 0.0004, 0.005, 0.0058]),
 DenseVector([-0.0046, 0.0017, 0.0011, 0.0033, 0.0069]),
 DenseVector([-0.0004, 0.0054, 0.0015, 0.0017, 0.0057]),
 DenseVector([-0.0, 0.0024, 0.0006, 0.0015, 0.0017]),
 DenseVector([-0.0003, 0.0005, 0.0001, 0.0055, 0.0052]),
 DenseVector([-0.0007, 0.0036, 0.001, 0.0025, 0.0051]),
 DenseVector([-0.0032, 0.0086, 0.0028, -0.0013, 0.0074]),
 DenseVector([-0.0006, 0.0005, 0.0001, 0.0066, 0.007]),
 DenseVector([-0.0, 0.0025, 0.0006, 0.0059, 0.0078]),
 DenseVector([-0.0, 0.0019, 0.0005, 0.0036, 0.0041]),
 DenseVector([-0.0002, 0.003, 0.0008, 0.0025, 0.0039]),
 DenseVector([-0.0004, 0.0015, 0.0004, 0.006, 0.0072]),
 DenseVector([-0.0004, 0.0047, 0.0013, 0.0018, 0.0051]),
 DenseVector([-0.0013, 0.0006, 0.0002, 0.0076, 0.0088]),
 DenseVector([-0.0036, -0.0003, 0.0005, 0.0027, 0.0032]),
 DenseVector([-0.0002, 0.0047, 0.0013, 0.0015, 0.0045]),
 DenseVector([-0.0007, 0.0043, 0.0012, 0.0021, 0.0053]),
 DenseVector([-0.0001, 0.0082, 0.0022, 0.0011, 0.0079]),
 DenseVector([-0.0005, 0.0014, 0.0004, 0.0038, 0.004]),
 DenseVector([-0.0004, 0.0017, 0.0005, 0.004, 0.0047]),
 DenseVector([-0.001, 0.0005, 0.0002, 0.0054, 0.0055]),
 DenseVector([-0.0003, 0.002, 0.0005, 0.0045, 0.0055]),
 DenseVector([-0.0, 0.0012, 0.0003, 0.0037, 0.0033]),
 DenseVector([-0.0009, 0.0018, 0.0006, 0.0037, 0.0047]),
 DenseVector([0.0005, 0.0084, 0.0022, -0.0007, 0.0052]),
 DenseVector([-0.0003, 0.0014, 0.0004, 0.0049, 0.0054]),
 DenseVector([-0.0, 0.0062, 0.0017, 0.0009, 0.0053]),
 DenseVector([-0.0, 0.0076, 0.002, 0.0001, 0.0057]),
 DenseVector([-0.0, 0.0129, 0.0035, -0.0046, 0.0055]),
 DenseVector([-0.0015, -0.0003, 0.0, 0.0079, 0.0084]),
 DenseVector([-0.0002, 0.0016, 0.0003, 0.007, 0.0083]),
 DenseVector([-0.0003, 0.0012, 0.0003, 0.0045, 0.0046]),
 DenseVector([-0.0005, 0.0018, 0.0004, 0.0071, 0.0089]),
 DenseVector([0.0003, 0.0027, 0.0006, 0.0039, 0.005]),
 DenseVector([-0.0001, 0.0032, 0.0008, 0.0047, 0.007]),
 DenseVector([-0.0008, 0.0008, 0.0003, 0.0047, 0.0048]),
 DenseVector([-0.0, 0.0026, 0.0006, 0.0034, 0.0045]),
 DenseVector([-0.0001, 0.0083, 0.0022, 0.0006, 0.0073]),
 DenseVector([0.0001, 0.0008, 0.0001, 0.0056, 0.0054]),
 DenseVector([-0.0004, 0.0041, 0.0011, 0.0045, 0.0079]),
 DenseVector([-0.0001, 0.0021, 0.0005, 0.0069, 0.0087]),
 DenseVector([-0.0, 0.0001, -0.0, 0.0045, 0.0032]),
 DenseVector([-0.0005, 0.0029, 0.0008, 0.0024, 0.0039]),
 DenseVector([-0.0005, 0.0015, 0.0004, 0.0055, 0.0066]),
 DenseVector([-0.0005, -0.0001, -0.0001, 0.0073, 0.0071]),
 DenseVector([-0.0019, 0.0032, 0.0012, 0.0011, 0.0037]),
 DenseVector([-0.001, 0.0019, 0.0006, 0.004, 0.0052]),
 DenseVector([-0.0001, 0.0013, 0.0003, 0.0045, 0.0046]),
 DenseVector([-0.0007, 0.0098, 0.0027, -0.0004, 0.0081]),
 DenseVector([-0.0006, 0.0029, 0.0009, 0.0011, 0.0021]),
 DenseVector([-0.0, 0.0017, 0.0003, 0.0067, 0.0079]),
 DenseVector([-0.0, 0.0049, 0.0013, 0.0012, 0.004]),
 DenseVector([-0.0, 0.0048, 0.0012, 0.0044, 0.0084]),
 DenseVector([-0.0, 0.0027, 0.0007, 0.0025, 0.0035]),
 DenseVector([-0.0001, 0.0027, 0.0007, 0.0015, 0.0021]),
 DenseVector([-0.0, 0.0017, 0.0004, 0.0049, 0.0056]),
 DenseVector([-0.0011, 0.0006, 0.0002, 0.0074, 0.0085]),
 DenseVector([-0.0006, 0.0013, 0.0004, 0.005, 0.0056]),
 DenseVector([-0.0008, 0.0034, 0.0009, 0.0044, 0.0074]),
 DenseVector([-0.0003, 0.0033, 0.0008, 0.0057, 0.0086]),
 DenseVector([-0.0001, 0.0029, 0.0007, 0.0031, 0.0045]),
 DenseVector([-0.0004, 0.0027, 0.0007, 0.0055, 0.0078]),
 DenseVector([-0.0017, 0.0022, 0.0008, 0.0047, 0.0071]),
 DenseVector([-0.0004, 0.001, 0.0002, 0.007, 0.0079]),
 DenseVector([-0.0001, 0.0014, 0.0003, 0.007, 0.0081]),
 DenseVector([-0.0002, 0.0025, 0.0006, 0.0045, 0.006]),
 DenseVector([-0.0, 0.0014, 0.0003, 0.0053, 0.0058]),
 DenseVector([-0.0005, 0.0013, 0.0003, 0.007, 0.0083]),
 DenseVector([-0.0012, 0.0011, 0.0004, 0.0065, 0.0079]),
 DenseVector([-0.0002, 0.0042, 0.001, 0.0047, 0.0082]),
 DenseVector([-0.0006, 0.0027, 0.0008, 0.0031, 0.0046]),
 DenseVector([-0.0003, 0.0017, 0.0005, 0.0035, 0.0039]),
 DenseVector([0.0005, 0.005, 0.0012, 0.0031, 0.0065]),
 DenseVector([-0.0008, 0.0058, 0.0017, 0.0002, 0.0045]),
 DenseVector([-0.0, 0.0049, 0.0012, 0.0036, 0.0074]),
 DenseVector([-0.0, 0.0019, 0.0004, 0.0046, 0.0052]),
 DenseVector([-0.0007, 0.005, 0.0014, 0.0018, 0.0057]),
 DenseVector([-0.0, 0.0024, 0.0005, 0.0057, 0.0074]),
 DenseVector([-0.0006, 0.0031, 0.0008, 0.0045, 0.007]),
 DenseVector([-0.0002, 0.003, 0.0008, 0.0046, 0.0068]),
 DenseVector([-0.0036, 0.0008, 0.0008, 0.0021, 0.0036]),
 DenseVector([-0.0001, 0.0026, 0.0007, 0.0026, 0.0035]),
 DenseVector([-0.0002, 0.0015, 0.0003, 0.0069, 0.0081]),
 DenseVector([-0.0006, 0.0029, 0.0008, 0.0033, 0.0052]),
 DenseVector([-0.0012, 0.0039, 0.0012, 0.0032, 0.0066]),
 DenseVector([-0.0003, 0.0029, 0.0007, 0.006, 0.0085]),
 DenseVector([-0.0004, 0.0031, 0.0009, 0.0034, 0.0054]),
 DenseVector([-0.0, 0.009, 0.0024, -0.0002, 0.0069]),
 DenseVector([-0.0001, 0.0057, 0.0015, 0.0016, 0.0057]),
 DenseVector([0.0006, 0.0022, 0.0004, 0.0032, 0.0033]),
 DenseVector([-0.002, 0.0017, 0.0007, 0.0061, 0.0086]),
 DenseVector([-0.0347, -0.0071, 0.004, -0.0077, 0.0054]),
 DenseVector([-0.0, 0.0018, 0.0004, 0.0058, 0.0069]),
 DenseVector([-0.0014, 0.0015, 0.0006, 0.0043, 0.0056]),
 DenseVector([-0.0001, 0.0021, 0.0005, 0.0062, 0.0078]),
 DenseVector([-0.0, 0.0054, 0.0014, 0.0032, 0.0074]),
 DenseVector([-0.0002, 0.0028, 0.0008, 0.0025, 0.0037]),
 DenseVector([-0.0012, 0.0016, 0.0006, 0.0042, 0.0054]),
 DenseVector([-0.001, 0.0014, 0.0005, 0.0048, 0.0057]),
 DenseVector([0.0005, 0.0026, 0.0006, 0.0042, 0.0052]),
 DenseVector([-0.0022, 0.0037, 0.0013, 0.0032, 0.0072]),
 DenseVector([-0.0001, 0.0005, 0.0001, 0.0048, 0.004]),
 DenseVector([-0.0, 0.0068, 0.0018, 0.0024, 0.0079]),
 DenseVector([-0.0001, 0.0013, 0.0003, 0.0057, 0.0061]),
 DenseVector([0.0001, 0.0092, 0.0025, -0.0016, 0.0052]),
 DenseVector([-0.0007, 0.0054, 0.0015, 0.0018, 0.006]),
 DenseVector([-0.001, 0.0021, 0.0006, 0.006, 0.0081]),
 DenseVector([-0.0003, 0.0015, 0.0003, 0.0056, 0.0064]),
 DenseVector([-0.0001, 0.0108, 0.0029, -0.0022, 0.0064]),
 DenseVector([-0.0008, 0.0024, 0.0007, 0.0028, 0.004]),
 DenseVector([-0.0, 0.0026, 0.0007, 0.0033, 0.0045]),
 DenseVector([-0.0001, 0.0047, 0.0012, 0.005, 0.0089]),
 DenseVector([-0.0, 0.0028, 0.0007, 0.0025, 0.0035]),
 DenseVector([-0.0001, 0.0054, 0.0014, 0.0021, 0.006]),
 DenseVector([-0.0015, 0.0048, 0.0016, 0.0002, 0.0039]),
 DenseVector([-0.0003, 0.0028, 0.0008, 0.0036, 0.0052]),
 DenseVector([-0.0005, 0.0012, 0.0003, 0.0057, 0.0064]),
 DenseVector([-0.0004, 0.0046, 0.0012, 0.0032, 0.0068]),
 DenseVector([-0.0018, 0.001, 0.0005, 0.0046, 0.0051]),
 DenseVector([-0.0, 0.0012, 0.0003, 0.0033, 0.0021]),
 DenseVector([-0.0002, 0.0051, 0.0013, 0.003, 0.0063]),
 DenseVector([-0.0003, 0.0025, 0.0006, 0.0066, 0.0084]),
 DenseVector([-0.0012, 0.0018, 0.0006, 0.0048, 0.0059]),
 DenseVector([-0.0004, 0.0013, 0.0003, 0.0046, 0.0043]),
 DenseVector([-0.0, 0.0041, 0.0011, 0.0031, 0.0054]),
 DenseVector([-0.0001, 0.0029, 0.0008, 0.0034, 0.0044]),
 DenseVector([-0.0002, 0.0007, 0.0002, 0.0031, 0.0014]),
 DenseVector([-0.0, 0.0046, 0.0012, 0.0014, 0.0036]),
 DenseVector([-0.0001, 0.0026, 0.0007, 0.0036, 0.0044]),
 DenseVector([-0.0019, 0.0082, 0.0025, -0.0021, 0.0043]),
 DenseVector([-0.0002, 0.0038, 0.001, 0.0007, 0.0019]),
 DenseVector([-0.0002, 0.0025, 0.0006, 0.0051, 0.0063]),
 DenseVector([-0.0002, 0.0011, 0.0002, 0.0064, 0.0065]),
 DenseVector([-0.0006, 0.0023, 0.0007, 0.0036, 0.0043]),
 DenseVector([-0.0001, 0.0031, 0.0008, 0.0039, 0.0054]),
 DenseVector([-0.0, 0.0039, 0.001, 0.0019, 0.0034]),
 DenseVector([-0.0003, 0.0031, 0.0008, 0.0056, 0.0077]),
 DenseVector([-0.0011, 0.0015, 0.0005, 0.004, 0.0043]),
 DenseVector([-0.0004, 0.0042, 0.0011, 0.0038, 0.0065]),
 DenseVector([-0.012, -0.0004, 0.0019, -0.0006, 0.0044]),
 DenseVector([-0.0005, 0.0057, 0.0016, 0.0015, 0.0054]),
 DenseVector([-0.0, 0.0054, 0.0014, 0.0013, 0.0044]),
 DenseVector([-0.0024, 0.0004, 0.0004, 0.007, 0.0082]),
 DenseVector([-0.0005, 0.0016, 0.0004, 0.007, 0.0081]),
 DenseVector([-0.0, 0.0067, 0.0018, -0.0005, 0.0032]),
 DenseVector([-0.0012, 0.0157, 0.0045, -0.0053, 0.0081]),
 DenseVector([-0.0001, 0.0034, 0.0008, 0.0058, 0.0083]),
 DenseVector([-0.0008, 0.0136, 0.0038, -0.0027, 0.0088]),
 DenseVector([-0.0123, -0.002, 0.0015, 0.0001, 0.0039]),
 DenseVector([-0.0007, 0.0014, 0.0004, 0.0055, 0.0059]),
 DenseVector([-0.0, 0.0033, 0.0009, 0.0015, 0.0023]),
 DenseVector([-0.0017, 0.0098, 0.0029, -0.0025, 0.0054]),
 DenseVector([-0.0, 0.0018, 0.0004, 0.0035, 0.0031]),
 DenseVector([-0.0004, 0.003, 0.0008, 0.0034, 0.0048]),
 DenseVector([-0.0, 0.0011, 0.0002, 0.005, 0.0044]),
 DenseVector([-0.0004, 0.0026, 0.0007, 0.0029, 0.0036]),
 DenseVector([-0.0003, 0.002, 0.0005, 0.0053, 0.0061]),
 DenseVector([-0.0, 0.0017, 0.0004, 0.0051, 0.0052]),
 DenseVector([-0.0002, 0.0057, 0.0015, 0.0025, 0.0064]),
 DenseVector([-0.0, 0.0006, 0.0001, 0.0041, 0.0026]),
 DenseVector([-0.0003, 0.002, 0.0005, 0.0057, 0.0067]),
 DenseVector([-0.0, 0.0028, 0.0006, 0.0056, 0.0071]),
 DenseVector([-0.0, 0.0044, 0.0011, 0.0039, 0.0068]),
 DenseVector([-0.0, 0.0053, 0.0013, 0.0043, 0.0083]),
 DenseVector([-0.0027, 0.0024, 0.001, 0.0034, 0.0058]),
 DenseVector([-0.0004, 0.0064, 0.0018, 0.0003, 0.0044]),
 DenseVector([-0.0024, 0.0027, 0.0011, 0.0033, 0.0059]),
 DenseVector([-0.0006, 0.0029, 0.0008, 0.0023, 0.0032]),
 DenseVector([-0.0004, 0.0089, 0.0024, 0.0001, 0.007]),
 DenseVector([0.0, 0.0078, 0.0021, -0.0004, 0.0047]),
 DenseVector([-0.0003, 0.0024, 0.0006, 0.0052, 0.0063]),
 DenseVector([-0.0017, 0.0017, 0.0007, 0.0032, 0.0039]),
 DenseVector([-0.0001, 0.0012, 0.0003, 0.0042, 0.0036]),
 DenseVector([-0.0001, 0.0068, 0.0018, 0.0005, 0.0048]),
 DenseVector([-0.0, 0.0033, 0.0008, 0.0039, 0.0054]),
 DenseVector([-0.0001, 0.0025, 0.0006, 0.0057, 0.0071]),
 DenseVector([-0.0001, 0.0083, 0.0023, -0.0018, 0.0035]),
 DenseVector([-0.0003, 0.0066, 0.0018, 0.0018, 0.0065]),
 DenseVector([-0.0002, 0.0015, 0.0004, 0.0036, 0.0032]),
 DenseVector([-0.0002, 0.0046, 0.0012, 0.0021, 0.0046]),
 DenseVector([-0.0, 0.003, 0.0008, 0.0036, 0.0048]),
 DenseVector([-0.0, 0.0025, 0.0006, 0.0038, 0.0043]),
 DenseVector([-0.0005, 0.0025, 0.0007, 0.0042, 0.0053]),
 DenseVector([-0.0016, 0.0032, 0.0011, 0.0024, 0.0046]),
 DenseVector([-0.003, 0.0001, 0.0004, 0.0063, 0.0074]),
 DenseVector([-0.0061, -0.0008, 0.0008, 0.0029, 0.0042]),
 DenseVector([-0.0003, 0.0025, 0.0007, 0.0038, 0.0048]),
 DenseVector([-0.0002, 0.0099, 0.0027, -0.0, 0.0078]),
 DenseVector([-0.0024, 0.0095, 0.003, -0.0028, 0.0052]),
 DenseVector([-0.0, 0.0035, 0.0009, 0.003, 0.0046]),
 DenseVector([-0.0001, 0.0016, 0.0004, 0.0047, 0.0048]),
 DenseVector([-0.0005, 0.0025, 0.0007, 0.0026, 0.0032]),
 DenseVector([-0.0003, 0.002, 0.0005, 0.0065, 0.0077]),
 DenseVector([-0.0001, 0.0033, 0.0008, 0.0041, 0.0058]),
 DenseVector([-0.0008, 0.0022, 0.0007, 0.0031, 0.0038]),
 DenseVector([-0.0007, 0.0025, 0.0007, 0.0063, 0.0083]),
 DenseVector([-0.0008, 0.0018, 0.0006, 0.0035, 0.0038]),
 DenseVector([-0.0001, 0.0033, 0.0008, 0.0047, 0.0065]),
 DenseVector([-0.0003, 0.0014, 0.0004, 0.0039, 0.0036]),
 DenseVector([-0.0002, 0.0041, 0.0011, 0.002, 0.004]),
 DenseVector([-0.0, 0.0037, 0.001, 0.0008, 0.0017]),
 DenseVector([-0.0005, 0.0017, 0.0005, 0.0035, 0.0036]),
 DenseVector([-0.0003, 0.0018, 0.0004, 0.0061, 0.0069]),
 DenseVector([-0.0638, 0.001, 0.0114, -0.0323, 0.0034]),
 DenseVector([0.0001, 0.0014, 0.0003, 0.0065, 0.0067]),
 DenseVector([-0.0004, 0.0018, 0.0005, 0.0032, 0.0032]),
 DenseVector([-0.0002, 0.0032, 0.0008, 0.0037, 0.0052]),
 DenseVector([-0.0011, 0.0044, 0.0013, 0.0032, 0.0067]),
 DenseVector([-0.0001, 0.0011, 0.0002, 0.0052, 0.0047]),
 DenseVector([-0.0, 0.0025, 0.0006, 0.0049, 0.0059]),
 DenseVector([-0.0003, 0.006, 0.0016, 0.0006, 0.0042]),
 DenseVector([-0.0004, 0.002, 0.0006, 0.0033, 0.0036]),
 DenseVector([-0.0004, 0.0054, 0.0015, 0.002, 0.0056]),
 DenseVector([0.0001, 0.0024, 0.0005, 0.0043, 0.0048]),
 DenseVector([-0.0, 0.0082, 0.0022, -0.0011, 0.0043]),
 DenseVector([0.0001, 0.0033, 0.0008, 0.0024, 0.0034]),
 DenseVector([-0.0001, 0.0019, 0.0005, 0.0037, 0.0037]),
 DenseVector([0.0, 0.0016, 0.0003, 0.0074, 0.0083]),
 DenseVector([-0.0, 0.0035, 0.0009, 0.003, 0.0046]),
 DenseVector([0.0007, 0.0019, 0.0003, 0.0054, 0.0054]),
 DenseVector([-0.0006, 0.0006, 0.0002, 0.0052, 0.0045]),
 DenseVector([-0.0001, 0.0018, 0.0004, 0.004, 0.004]),
 DenseVector([-0.0023, 0.002, 0.0008, 0.0056, 0.0079]),
 DenseVector([-0.0, 0.0025, 0.0006, 0.0035, 0.004]),
 DenseVector([-0.0001, 0.0091, 0.0024, -0.0, 0.0068]),
 DenseVector([-0.0001, 0.0025, 0.0006, 0.0062, 0.0077]),
 DenseVector([-0.0004, 0.0021, 0.0005, 0.0048, 0.0055]),
 DenseVector([-0.0006, 0.0002, 0.0001, 0.0062, 0.0055]),
 DenseVector([-0.0002, 0.0093, 0.0026, -0.0017, 0.0046]),
 DenseVector([-0.0015, 0.0001, 0.0002, 0.006, 0.0059]),
 DenseVector([-0.0014, 0.0009, 0.0004, 0.0044, 0.0045]),
 DenseVector([-0.0013, 0.0221, 0.0063, -0.0137, 0.0039]),
 DenseVector([-0.001, 0.0035, 0.0011, 0.0027, 0.0047]),
 DenseVector([-0.0014, 0.0039, 0.0012, 0.0035, 0.0067]),
 DenseVector([-0.0004, 0.001, 0.0003, 0.0038, 0.003]),
 DenseVector([-0.0002, 0.0041, 0.0011, 0.0022, 0.0042]),
 DenseVector([-0.0027, 0.0017, 0.0008, 0.0031, 0.0045]),
 DenseVector([-0.0016, 0.0002, 0.0002, 0.0066, 0.0068]),
 DenseVector([-0.0012, 0.0016, 0.0006, 0.0042, 0.0049]),
 DenseVector([-0.0011, 0.0024, 0.0008, 0.0036, 0.0049]),
 DenseVector([-0.0, 0.0025, 0.0006, 0.0044, 0.0054]),
 DenseVector([-0.0003, 0.0003, 0.0001, 0.0043, 0.0027]),
 DenseVector([-0.0, 0.0064, 0.0016, 0.0029, 0.0076]),
 DenseVector([-0.0002, 0.006, 0.0016, 0.0015, 0.0054]),
 DenseVector([0.0001, 0.0036, 0.0008, 0.0059, 0.0085]),
 DenseVector([-0.0003, 0.0022, 0.0006, 0.0052, 0.0061]),
 DenseVector([0.0001, 0.0039, 0.001, 0.0029, 0.0048]),
 DenseVector([-0.0, 0.0024, 0.0006, 0.003, 0.0032]),
 DenseVector([-0.0, 0.0029, 0.0008, 0.0021, 0.0026]),
 DenseVector([-0.0, 0.0045, 0.0012, 0.0018, 0.004]),
 DenseVector([-0.0075, 0.0013, 0.0016, 0.0028, 0.0075]),
 DenseVector([0.0003, 0.0026, 0.0006, 0.0027, 0.0028]),
 DenseVector([-0.0, 0.0024, 0.0006, 0.004, 0.0045]),
 DenseVector([-0.0001, 0.003, 0.0008, 0.0034, 0.0045]),
 DenseVector([-0.0001, 0.0026, 0.0006, 0.0054, 0.0067]),
 DenseVector([0.0001, 0.0016, 0.0003, 0.0055, 0.0055]),
 DenseVector([-0.0, 0.0022, 0.0005, 0.0038, 0.0042]),
 DenseVector([-0.0004, 0.002, 0.0005, 0.0051, 0.0059]),
 DenseVector([0.0002, 0.0007, 0.0001, 0.0045, 0.0032]),
 DenseVector([-0.0003, 0.0049, 0.0014, 0.0014, 0.0042]),
 DenseVector([-0.0, 0.0047, 0.0013, 0.0006, 0.0027]),
 DenseVector([-0.0013, 0.004, 0.0012, 0.0022, 0.005]),
 DenseVector([-0.0018, 0.0026, 0.0009, 0.0036, 0.0057]),
 DenseVector([0.0004, 0.0073, 0.0018, 0.0028, 0.0082]),
 DenseVector([-0.0001, 0.0021, 0.0005, 0.0055, 0.0063]),
 DenseVector([-0.0, 0.0011, 0.0002, 0.0081, 0.0086]),
 DenseVector([-0.001, 0.0028, 0.0009, 0.0012, 0.002]),
 DenseVector([-0.0077, -0.0009, 0.001, 0.005, 0.0081]),
 DenseVector([-0.0007, 0.0023, 0.0007, 0.0029, 0.0036]),
 DenseVector([-0.0006, 0.0122, 0.0034, -0.002, 0.0079]),
 DenseVector([-0.0, 0.0032, 0.0008, 0.0036, 0.005]),
 DenseVector([-0.0024, 0.0004, 0.0004, 0.0058, 0.0066]),
 DenseVector([-0.0002, 0.0026, 0.0007, 0.0052, 0.0064]),
 DenseVector([-0.0001, 0.0022, 0.0005, 0.0049, 0.0055]),
 DenseVector([-0.0002, 0.0028, 0.0007, 0.0043, 0.0056]),
 DenseVector([-0.0, 0.0016, 0.0004, 0.0035, 0.003]),
 DenseVector([-0.0014, 0.0014, 0.0006, 0.0033, 0.0036]),
 DenseVector([-0.0022, 0.0029, 0.0011, 0.0033, 0.0059]),
 DenseVector([-0.0011, 0.0018, 0.0006, 0.0043, 0.0051]),
 DenseVector([-0.0007, 0.0015, 0.0004, 0.0062, 0.0071]),
 DenseVector([-0.0, 0.0029, 0.0007, 0.0022, 0.0026]),
 DenseVector([-0.0002, 0.0036, 0.0009, 0.005, 0.0073]),
 DenseVector([-0.0019, 0.0044, 0.0015, 0.0018, 0.0053]),
 DenseVector([-0.0007, 0.0037, 0.0011, 0.0024, 0.0044]),
 DenseVector([-0.0007, 0.0028, 0.0008, 0.0043, 0.0059]),
 DenseVector([-0.0008, 0.0055, 0.0016, 0.0023, 0.0064]),
 DenseVector([-0.0001, 0.0041, 0.0011, 0.0012, 0.0028]),
 DenseVector([-0.0003, 0.0021, 0.0005, 0.0048, 0.0055]),
 DenseVector([-0.0003, 0.008, 0.0022, -0.0019, 0.0031]),
 DenseVector([-0.0007, 0.0025, 0.0007, 0.004, 0.0051]),
 DenseVector([-0.0055, 0.0002, 0.0009, 0.0027, 0.0046]),
 DenseVector([-0.0018, 0.0028, 0.001, 0.0045, 0.0071]),
 DenseVector([-0.0011, 0.0021, 0.0007, 0.0045, 0.0057]),
 DenseVector([-0.0002, 0.001, 0.0002, 0.0063, 0.0063]),
 DenseVector([0.0001, 0.0036, 0.0009, 0.0039, 0.0057]),
 DenseVector([-0.0003, 0.0001, 0.0, 0.005, 0.0036]),
 DenseVector([0.0002, 0.0012, 0.0002, 0.0044, 0.0035]),
 DenseVector([-0.0001, 0.0056, 0.0015, 0.0023, 0.006]),
 DenseVector([-0.0007, 0.002, 0.0006, 0.0054, 0.0065]),
 DenseVector([-0.0004, 0.0017, 0.0004, 0.006, 0.0067]),
 DenseVector([-0.0, 0.002, 0.0005, 0.0036, 0.0036]),
 DenseVector([-0.0, 0.0006, 0.0001, 0.0057, 0.0047]),
 DenseVector([-0.0009, 0.0023, 0.0007, 0.0031, 0.004]),
 DenseVector([-0.0002, 0.0003, -0.0, 0.0085, 0.0084]),
 DenseVector([-0.0001, 0.0034, 0.0008, 0.004, 0.0058]),
 DenseVector([-0.0002, 0.0019, 0.0005, 0.0039, 0.004]),
 DenseVector([-0.0007, 0.0054, 0.0015, 0.0036, 0.0079]),
 DenseVector([-0.0005, 0.0015, 0.0004, 0.0049, 0.0051]),
 DenseVector([-0.0001, 0.0015, 0.0004, 0.0034, 0.0028]),
 DenseVector([-0.0001, 0.0039, 0.001, 0.0015, 0.0029]),
 DenseVector([-0.0013, 0.0026, 0.0009, 0.0042, 0.0061]),
 DenseVector([0.0, 0.0017, 0.0004, 0.0037, 0.0034]),
 DenseVector([0.0001, 0.0018, 0.0004, 0.0041, 0.0039]),
 DenseVector([-0.0007, 0.002, 0.0006, 0.004, 0.0045]),
 DenseVector([-0.0011, 0.0017, 0.0005, 0.0064, 0.0078]),
 DenseVector([-0.0003, 0.0035, 0.0009, 0.0029, 0.0046]),
 DenseVector([-0.0005, 0.0034, 0.0009, 0.0034, 0.0051]),
 DenseVector([-0.0045, -0.0011, 0.0004, 0.0061, 0.007]),
 DenseVector([-0.0002, 0.002, 0.0005, 0.0038, 0.004]),
 DenseVector([-0.0003, 0.0056, 0.0015, 0.0006, 0.0038]),
 DenseVector([-0.0001, 0.0024, 0.0006, 0.0051, 0.0061]),
 DenseVector([-0.0, 0.0014, 0.0002, 0.0078, 0.0084]),
 DenseVector([-0.0077, 0.0027, 0.002, 0.0018, 0.0079]),
 DenseVector([-0.0025, 0.0051, 0.0018, 0.0001, 0.0044]),
 DenseVector([0.0002, 0.0004, -0.0, 0.0068, 0.0059]),
 DenseVector([-0.0, 0.0026, 0.0007, 0.0031, 0.0036]),
 DenseVector([-0.0072, 0.015, 0.0054, -0.0122, 0.0025]),
 DenseVector([-0.0008, 0.0009, 0.0003, 0.0046, 0.0043]),
 DenseVector([-0.0023, 0.005, 0.0017, 0.0029, 0.0079]),
 DenseVector([0.0002, 0.0026, 0.0006, 0.0042, 0.005]),
 DenseVector([-0.0, 0.0027, 0.0006, 0.0057, 0.0071]),
 DenseVector([-0.0003, 0.0024, 0.0006, 0.006, 0.0075]),
 DenseVector([-0.0003, 0.0024, 0.0005, 0.0071, 0.0089]),
 DenseVector([-0.0004, 0.0008, 0.0002, 0.0071, 0.0073]),
 DenseVector([-0.0, 0.0014, 0.0003, 0.0059, 0.0061]),
 DenseVector([-0.0, 0.0036, 0.0009, 0.005, 0.0072]),
 DenseVector([-0.0003, 0.007, 0.0019, 0.0011, 0.006]),
 DenseVector([-0.0003, 0.0012, 0.0003, 0.0048, 0.0045]),
 DenseVector([-0.0001, 0.0021, 0.0005, 0.0066, 0.0078]),
 DenseVector([-0.0032, 0.0012, 0.0008, 0.0055, 0.0076]),
 DenseVector([-0.0, 0.0047, 0.0012, 0.0028, 0.0056]),
 DenseVector([-0.0, 0.0011, 0.0002, 0.0049, 0.0041]),
 DenseVector([-0.0, 0.0021, 0.0005, 0.0037, 0.0038]),
 DenseVector([-0.0012, 0.0038, 0.0011, 0.0044, 0.0077]),
 DenseVector([-0.0005, 0.0051, 0.0014, 0.0012, 0.0042]),
 DenseVector([-0.0, 0.005, 0.0013, 0.0006, 0.0029]),
 DenseVector([0.0005, 0.0017, 0.0003, 0.0046, 0.0041]),
 DenseVector([0.0, 0.0046, 0.0012, 0.0018, 0.004]),
 DenseVector([-0.0008, 0.0008, 0.0003, 0.0047, 0.0043]),
 DenseVector([-0.0057, 0.0, 0.0009, 0.0029, 0.0048]),
 DenseVector([-0.0001, 0.0042, 0.0011, 0.0037, 0.0064]),
 DenseVector([-0.0002, 0.0035, 0.0009, 0.0053, 0.0077]),
 DenseVector([-0.0001, 0.0023, 0.0006, 0.0024, 0.0024]),
 DenseVector([-0.0, 0.0019, 0.0004, 0.0066, 0.0074]),
 DenseVector([-0.0007, 0.0069, 0.002, -0.0014, 0.0029]),
 DenseVector([-0.0004, 0.0017, 0.0004, 0.0056, 0.0063]),
 DenseVector([-0.0, 0.0005, 0.0001, 0.0045, 0.0028]),
 DenseVector([-0.0006, 0.0254, 0.0071, -0.0155, 0.0047]),
 DenseVector([-0.0006, 0.0028, 0.0008, 0.0031, 0.0041]),
 DenseVector([0.0002, 0.0021, 0.0005, 0.0041, 0.0042]),
 DenseVector([-0.0, 0.0055, 0.0014, 0.0039, 0.0079]),
 DenseVector([-0.0001, 0.0016, 0.0003, 0.0063, 0.0069]),
 DenseVector([-0.0004, 0.0027, 0.0007, 0.0035, 0.0046]),
 DenseVector([-0.0, 0.0013, 0.0002, 0.0064, 0.0065]),
 DenseVector([-0.0009, 0.0025, 0.0007, 0.0051, 0.0069]),
 DenseVector([-0.0013, 0.0022, 0.0007, 0.004, 0.0053]),
 DenseVector([-0.0003, 0.0018, 0.0005, 0.0033, 0.0032]),
 DenseVector([-0.0006, 0.0007, 0.0002, 0.0036, 0.0026]),
 DenseVector([-0.0019, 0.004, 0.0013, 0.0029, 0.0063]),
 DenseVector([-0.0004, 0.0037, 0.001, 0.005, 0.0077]),
 DenseVector([-0.0048, 0.0016, 0.0012, 0.0039, 0.0072]),
 DenseVector([-0.003, 0.0013, 0.0008, 0.0043, 0.0061]),
 DenseVector([-0.0041, 0.0024, 0.0013, 0.0016, 0.0045]),
 DenseVector([-0.0001, 0.0052, 0.0014, 0.0005, 0.0031]),
 DenseVector([-0.0031, 0.0011, 0.0008, 0.0032, 0.0045]),
 DenseVector([-0.0041, 0.0007, 0.0008, 0.0032, 0.0046]),
 DenseVector([0.0, 0.0011, 0.0002, 0.0053, 0.0047]),
 DenseVector([-0.0002, 0.0019, 0.0004, 0.0056, 0.0062]),
 DenseVector([-0.0, 0.0012, 0.0003, 0.0041, 0.0034]),
 DenseVector([-0.0002, 0.002, 0.0005, 0.0044, 0.0047]),
 DenseVector([0.0002, 0.004, 0.0009, 0.0057, 0.0085]),
 DenseVector([-0.0007, 0.0092, 0.0026, -0.0033, 0.0029]),
 DenseVector([-0.0026, 0.0015, 0.0008, 0.0021, 0.0029]),
 DenseVector([-0.0008, 0.0105, 0.003, -0.0037, 0.0039]),
 DenseVector([-0.0, 0.0051, 0.0013, 0.0039, 0.0075]),
 DenseVector([-0.0034, 0.0018, 0.001, 0.0018, 0.0034]),
 DenseVector([-0.0001, 0.0008, 0.0002, 0.0043, 0.0031]),
 DenseVector([-0.006, 0.0022, 0.0015, 0.0031, 0.0077]),
 DenseVector([-0.0006, 0.0031, 0.0009, 0.0021, 0.0031]),
 DenseVector([-0.0, 0.0031, 0.0008, 0.0038, 0.0052]),
 DenseVector([-0.0001, 0.0035, 0.0009, 0.0022, 0.0034]),
 DenseVector([0.0002, 0.0054, 0.0013, 0.0036, 0.0071]),
 DenseVector([-0.0009, 0.0024, 0.0007, 0.0053, 0.0071]),
 DenseVector([0.0003, 0.0021, 0.0004, 0.0063, 0.0071]),
 DenseVector([-0.0016, 0.0029, 0.001, 0.0014, 0.0028]),
 DenseVector([-0.0, 0.0032, 0.0008, 0.0041, 0.0054]),
 DenseVector([-0.0001, 0.0031, 0.0008, 0.0028, 0.0038]),
 DenseVector([-0.0003, 0.0043, 0.0011, 0.0045, 0.0077]),
 DenseVector([-0.0005, 0.0014, 0.0004, 0.0037, 0.0033]),
 DenseVector([-0.0003, 0.0032, 0.0009, 0.0028, 0.0042]),
 DenseVector([-0.0, 0.004, 0.001, 0.0031, 0.0052]),
 DenseVector([-0.0, 0.0005, 0.0, 0.0062, 0.0053]),
 DenseVector([-0.0013, 0.0014, 0.0005, 0.004, 0.0043]),
 DenseVector([-0.0002, 0.0071, 0.0019, -0.0002, 0.0042]),
 DenseVector([-0.0004, 0.0023, 0.0006, 0.0028, 0.0029]),
 DenseVector([-0.0001, 0.007, 0.0019, 0.0004, 0.005]),
 DenseVector([0.0005, 0.0006, -0.0, 0.0061, 0.0049]),
 DenseVector([-0.0003, 0.0041, 0.0011, 0.0011, 0.0026]),
 DenseVector([-0.0003, 0.0028, 0.0007, 0.0034, 0.0043]),
 DenseVector([-0.0, 0.0042, 0.0011, 0.0033, 0.0055]),
 DenseVector([-0.0002, 0.0055, 0.0015, 0.0014, 0.0046]),
 DenseVector([-0.0005, 0.0051, 0.0015, -0.0001, 0.0025]),
 DenseVector([0.0005, 0.0027, 0.0006, 0.0031, 0.0034]),
 DenseVector([-0.0, 0.0038, 0.001, 0.0031, 0.005]),
 DenseVector([0.0004, 0.0021, 0.0004, 0.0046, 0.0048]),
 DenseVector([0.0002, 0.0045, 0.0012, 0.001, 0.0027]),
 DenseVector([-0.0019, 0.0008, 0.0005, 0.0041, 0.0043]),
 DenseVector([-0.0004, 0.0029, 0.0008, 0.0023, 0.0032]),
 DenseVector([-0.0018, 0.0062, 0.002, 0.0006, 0.0058]),
 DenseVector([-0.0002, 0.0052, 0.0014, 0.0007, 0.0035]),
 DenseVector([-0.0012, 0.0088, 0.0026, -0.0027, 0.0037]),
 DenseVector([-0.0001, 0.0045, 0.0012, 0.0026, 0.005]),
 DenseVector([-0.0002, 0.0041, 0.0011, 0.0027, 0.0049]),
 DenseVector([-0.0002, 0.0017, 0.0004, 0.0039, 0.0037]),
 DenseVector([-0.0003, 0.0043, 0.0011, 0.0025, 0.0048]),
 DenseVector([-0.0001, 0.0027, 0.0006, 0.0068, 0.0086]),
 DenseVector([-0.0003, 0.0012, 0.0003, 0.006, 0.0061]),
 DenseVector([-0.0008, 0.0097, 0.0027, -0.0015, 0.006]),
 DenseVector([-0.0, 0.0037, 0.001, 0.0018, 0.003]),
 DenseVector([-0.0, 0.0035, 0.0009, 0.0046, 0.0065]),
 DenseVector([-0.0001, 0.0036, 0.001, 0.0015, 0.0026]),
 DenseVector([0.0002, 0.0054, 0.0014, 0.0005, 0.003]),
 DenseVector([-0.0006, 0.0034, 0.001, 0.0027, 0.0044]),
 DenseVector([-0.0003, 0.0041, 0.0011, 0.0022, 0.0041]),
 DenseVector([0.0002, 0.0063, 0.0017, 0.0003, 0.0038]),
 DenseVector([-0.0011, 0.004, 0.0012, 0.0009, 0.003]),
 DenseVector([-0.0, 0.0022, 0.0005, 0.0045, 0.0049]),
 DenseVector([-0.0001, 0.0033, 0.0008, 0.0042, 0.0059]),
 DenseVector([-0.0002, 0.0025, 0.0006, 0.004, 0.0047]),
 DenseVector([-0.0003, 0.0063, 0.0018, -0.0008, 0.0027]),
 DenseVector([-0.0001, 0.0042, 0.0011, 0.0039, 0.0066]),
 DenseVector([-0.0, 0.0009, 0.0002, 0.0057, 0.0051]),
 DenseVector([-0.0007, 0.0024, 0.0007, 0.0057, 0.0074]),
 DenseVector([-0.0004, 0.0029, 0.0008, 0.0035, 0.0047]),
 DenseVector([-0.0004, 0.0026, 0.0006, 0.0057, 0.0073]),
 DenseVector([-0.0001, 0.003, 0.0007, 0.0049, 0.0065]),
 DenseVector([-0.0, 0.0009, 0.0002, 0.0054, 0.0047]),
 DenseVector([-0.0013, 0.0006, 0.0003, 0.0033, 0.0025]),
 DenseVector([-0.0001, 0.0054, 0.0014, 0.0011, 0.004]),
 DenseVector([-0.001, 0.0048, 0.0014, 0.0012, 0.0041]),
 DenseVector([-0.0017, 0.0011, 0.0005, 0.0041, 0.0045]),
 DenseVector([-0.0006, 0.0212, 0.006, -0.0123, 0.0043]),
 DenseVector([-0.0023, 0.0039, 0.0014, 0.0022, 0.0055]),
 DenseVector([-0.0, 0.0008, 0.0001, 0.0055, 0.0047]),
 DenseVector([-0.0001, 0.0037, 0.0009, 0.0043, 0.0065]),
 DenseVector([0.0001, 0.0021, 0.0005, 0.003, 0.0028]),
 DenseVector([-0.003, 0.0024, 0.0011, 0.0048, 0.0079]),
 DenseVector([-0.0001, 0.0052, 0.0014, -0.0005, 0.0017]),
 DenseVector([-0.0006, 0.004, 0.0011, 0.0034, 0.0059]),
 DenseVector([-0.0, 0.0026, 0.0006, 0.003, 0.0033]),
 DenseVector([-0.0, 0.0017, 0.0004, 0.0052, 0.0053]),
 DenseVector([0.0007, 0.0034, 0.0007, 0.0042, 0.0053]),
 DenseVector([-0.0005, 0.0009, 0.0003, 0.0051, 0.0047]),
 DenseVector([-0.0001, 0.0002, -0.0, 0.0065, 0.0053]),
 DenseVector([-0.0003, 0.0037, 0.001, 0.0022, 0.0037]),
 DenseVector([-0.0007, 0.0013, 0.0004, 0.0057, 0.0061]),
 DenseVector([-0.0005, 0.0016, 0.0005, 0.0035, 0.0033]),
 DenseVector([-0.0, 0.002, 0.0005, 0.0032, 0.003]),
 DenseVector([-0.0, 0.0017, 0.0004, 0.0047, 0.0047]),
 DenseVector([-0.0002, 0.0012, 0.0003, 0.0045, 0.0039]),
 DenseVector([-0.0, 0.0005, -0.0, 0.0081, 0.0078]),
 DenseVector([-0.0006, 0.0013, 0.0004, 0.0047, 0.0047]),
 DenseVector([-0.0005, 0.0011, 0.0003, 0.0052, 0.0048]),
 DenseVector([-0.0005, 0.0078, 0.0021, 0.0013, 0.0074]),
 DenseVector([-0.0009, 0.0011, 0.0004, 0.0046, 0.0045]),
 DenseVector([-0.0008, 0.0024, 0.0007, 0.0042, 0.0055]),
 DenseVector([-0.0043, 0.0025, 0.0014, 0.0013, 0.0043]),
 DenseVector([-0.0001, 0.0026, 0.0006, 0.006, 0.0074]),
 DenseVector([-0.0001, 0.0023, 0.0005, 0.0068, 0.0081]),
 DenseVector([-0.0001, 0.0029, 0.0007, 0.0054, 0.007]),
 DenseVector([-0.0004, 0.0023, 0.0007, 0.0023, 0.0024]),
 DenseVector([-0.001, 0.002, 0.0007, 0.0033, 0.0039]),
 DenseVector([-0.0005, 0.0083, 0.0023, 0.0018, 0.0087]),
 DenseVector([-0.0001, 0.0024, 0.0006, 0.0045, 0.005]),
 DenseVector([-0.0001, 0.0004, 0.0, 0.006, 0.005]),
 DenseVector([-0.0002, 0.0204, 0.0056, -0.0103, 0.0058]),
 DenseVector([-0.0005, 0.0033, 0.0009, 0.0047, 0.0069]),
 DenseVector([-0.0003, 0.0026, 0.0007, 0.0051, 0.0065]),
 DenseVector([-0.0003, 0.0018, 0.0004, 0.0056, 0.0061]),
 DenseVector([-0.0003, 0.0015, 0.0004, 0.0044, 0.004]),
 DenseVector([-0.0002, 0.0042, 0.0011, 0.0017, 0.0036]),
 DenseVector([-0.0002, 0.008, 0.0022, -0.0012, 0.0038]),
 DenseVector([-0.0005, 0.0026, 0.0008, 0.0029, 0.0037]),
 DenseVector([-0.0, 0.0055, 0.0015, 0.0016, 0.0048]),
 DenseVector([-0.0, 0.0022, 0.0005, 0.0065, 0.0076]),
 DenseVector([-0.0002, 0.0006, 0.0, 0.0078, 0.0076]),
 DenseVector([-0.0008, 0.0005, 0.0002, 0.0045, 0.0037]),
 DenseVector([-0.0002, 0.0021, 0.0005, 0.0067, 0.008]),
 DenseVector([-0.0002, 0.0045, 0.0012, 0.0042, 0.0073]),
 DenseVector([-0.0007, 0.0029, 0.0008, 0.0034, 0.0049]),
 DenseVector([-0.0011, 0.0025, 0.0008, 0.0058, 0.008]),
 DenseVector([-0.0004, 0.0011, 0.0003, 0.0058, 0.0056]),
 DenseVector([-0.0017, 0.0023, 0.0008, 0.0037, 0.0052]),
 DenseVector([-0.0, 0.0041, 0.001, 0.0051, 0.0079]),
 DenseVector([-0.0004, 0.0027, 0.0007, 0.0025, 0.003]),
 DenseVector([-0.0001, 0.0017, 0.0004, 0.0042, 0.0041]),
 DenseVector([-0.0001, 0.0016, 0.0004, 0.0033, 0.0027]),
 DenseVector([-0.0002, 0.0009, 0.0002, 0.0039, 0.0027]),
 DenseVector([-0.0004, 0.0008, 0.0002, 0.0072, 0.0074]),
 DenseVector([-0.0002, 0.0022, 0.0006, 0.0036, 0.0039]),
 DenseVector([-0.0006, 0.0045, 0.0012, 0.0043, 0.0076]),
 DenseVector([-0.0008, 0.0015, 0.0005, 0.004, 0.0041]),
 DenseVector([-0.0, 0.0008, 0.0001, 0.0052, 0.0043]),
 DenseVector([-0.0053, 0.0017, 0.0013, 0.0028, 0.0061]),
 DenseVector([-0.0098, 0.0023, 0.0022, 0.0007, 0.0074]),
 DenseVector([-0.0004, 0.0033, 0.0008, 0.006, 0.0082]),
 DenseVector([0.0006, 0.0027, 0.0006, 0.0039, 0.004]),
 DenseVector([-0.0001, 0.0025, 0.0007, 0.003, 0.0033]),
 DenseVector([-0.0001, 0.003, 0.0007, 0.0042, 0.0055]),
 DenseVector([-0.0006, 0.0019, 0.0005, 0.0067, 0.008]),
 DenseVector([-0.0011, 0.0009, 0.0004, 0.0033, 0.0027]),
 DenseVector([-0.0003, 0.0025, 0.0007, 0.0042, 0.0051]),
 DenseVector([-0.0006, 0.0027, 0.0007, 0.0051, 0.0069]),
 DenseVector([-0.0002, 0.0012, 0.0003, 0.0057, 0.005]),
 DenseVector([-0.0004, 0.0006, 0.0001, 0.0063, 0.0053]),
 DenseVector([-0.0002, 0.0022, 0.0006, 0.0027, 0.002]),
 DenseVector([-0.0004, 0.0019, 0.0005, 0.0043, 0.004]),
 DenseVector([-0.0, 0.0018, 0.0004, 0.005, 0.0046]),
 DenseVector([-0.0002, 0.0015, 0.0003, 0.0062, 0.0061]),
 DenseVector([-0.0, 0.0037, 0.0009, 0.0034, 0.0046]),
 DenseVector([-0.0, 0.0017, 0.0004, 0.0046, 0.004]),
 DenseVector([-0.0001, 0.0016, 0.0003, 0.007, 0.0071]),
 DenseVector([-0.0001, 0.0064, 0.0017, 0.0017, 0.0055]),
 DenseVector([-0.0002, 0.0038, 0.001, 0.0032, 0.0046]),
 DenseVector([-0.0001, 0.0005, 0.0, 0.0065, 0.0051]),
 DenseVector([0.0006, 0.0046, 0.0011, 0.0032, 0.0049]),
 DenseVector([-0.0006, 0.0025, 0.0007, 0.004, 0.0046]),
 DenseVector([-0.0, 0.0023, 0.0006, 0.0029, 0.0021]),
 DenseVector([-0.0014, 0.0023, 0.0008, 0.0045, 0.0055]),
 DenseVector([0.0001, 0.003, 0.0007, 0.0054, 0.0063]),
 DenseVector([-0.0001, 0.0026, 0.0007, 0.0037, 0.0038]),
 DenseVector([-0.0023, 0.0006, 0.0004, 0.0068, 0.0075]),
 DenseVector([-0.0003, 0.0096, 0.0026, -0.0016, 0.0047]),
 DenseVector([-0.0003, 0.0025, 0.0007, 0.0028, 0.0027]),
 DenseVector([-0.0, 0.001, 0.0002, 0.0062, 0.0054]),
 DenseVector([-0.0003, 0.0018, 0.0005, 0.003, 0.002]),
 DenseVector([-0.0012, 0.0011, 0.0004, 0.0073, 0.0079]),
 DenseVector([-0.0004, 0.0052, 0.0015, 0.0005, 0.0027]),
 DenseVector([-0.0007, 0.0082, 0.0023, -0.0005, 0.0051]),
 DenseVector([-0.0006, 0.0066, 0.0018, 0.0004, 0.0043]),
 DenseVector([-0.0001, 0.006, 0.0016, -0.0001, 0.0026]),
 DenseVector([-0.0033, 0.0022, 0.0011, 0.0033, 0.0054]),
 DenseVector([-0.0002, 0.0017, 0.0004, 0.0051, 0.0048]),
 DenseVector([-0.0003, 0.0041, 0.0011, 0.0018, 0.0031]),
 DenseVector([-0.0004, 0.0036, 0.001, 0.0029, 0.0039]),
 DenseVector([-0.0004, 0.0015, 0.0004, 0.0033, 0.0023]),
 DenseVector([-0.0, 0.0045, 0.0011, 0.0048, 0.0073]),
 DenseVector([-0.0, 0.0015, 0.0004, 0.0033, 0.0021]),
 DenseVector([-0.0002, 0.0012, 0.0003, 0.0051, 0.0042]),
 DenseVector([-0.0003, 0.0017, 0.0004, 0.0041, 0.0034]),
 DenseVector([-0.0003, 0.0026, 0.0007, 0.0034, 0.0036]),
 DenseVector([-0.0, 0.0044, 0.0012, 0.0018, 0.0033]),
 DenseVector([-0.0, 0.0039, 0.001, 0.0022, 0.0033]),
 DenseVector([-0.0003, 0.0009, 0.0002, 0.0063, 0.0056]),
 DenseVector([-0.0006, 0.0027, 0.0008, 0.0023, 0.0025]),
 DenseVector([-0.0003, 0.0013, 0.0003, 0.0073, 0.0074]),
 DenseVector([-0.0003, 0.0004, 0.0, 0.0081, 0.0075]),
 DenseVector([0.0003, 0.0023, 0.0005, 0.0039, 0.0035]),
 DenseVector([0.0002, 0.0021, 0.0005, 0.0053, 0.0052]),
 DenseVector([-0.001, 0.0033, 0.001, 0.0038, 0.0056]),
 DenseVector([-0.0, 0.0014, 0.0003, 0.0034, 0.0021]),
 DenseVector([-0.0, 0.0115, 0.0031, -0.0008, 0.0077]),
 DenseVector([-0.0002, 0.0123, 0.0034, -0.0034, 0.0053]),
 DenseVector([-0.0006, 0.002, 0.0006, 0.0029, 0.0025]),
 DenseVector([-0.0, 0.0128, 0.0035, -0.0057, 0.0026]),
 DenseVector([-0.0002, 0.0055, 0.0015, 0.0011, 0.0037]),
 DenseVector([-0.0, 0.0024, 0.0006, 0.0028, 0.0023]),
 DenseVector([-0.0016, 0.002, 0.0007, 0.0063, 0.0077]),
 DenseVector([-0.0006, 0.0017, 0.0005, 0.0034, 0.0029]),
 DenseVector([-0.0014, 0.0035, 0.0011, 0.0023, 0.0041]),
 DenseVector([-0.0001, 0.0014, 0.0003, 0.004, 0.0029]),
 DenseVector([-0.0004, 0.0016, 0.0004, 0.0038, 0.0031]),
 DenseVector([0.0004, 0.0019, 0.0004, 0.0043, 0.0035]),
 DenseVector([-0.0004, 0.0017, 0.0004, 0.0042, 0.0037]),
 DenseVector([-0.0007, 0.0017, 0.0005, 0.0036, 0.0033]),
 DenseVector([-0.0001, 0.001, 0.0002, 0.0075, 0.0072]),
 DenseVector([-0.0012, 0.001, 0.0004, 0.0037, 0.003]),
 DenseVector([-0.0007, 0.0062, 0.0018, 0.0007, 0.0043]),
 DenseVector([-0.0, 0.0021, 0.0005, 0.0047, 0.0047]),
 DenseVector([-0.0, 0.0034, 0.0009, 0.0029, 0.0037]),
 DenseVector([-0.0006, 0.0021, 0.0006, 0.0044, 0.0046]),
 DenseVector([-0.0008, 0.0067, 0.0019, -0.0006, 0.0034]),
 DenseVector([0.0005, 0.0025, 0.0005, 0.005, 0.005]),
 DenseVector([0.0002, 0.0016, 0.0003, 0.0036, 0.0023]),
 DenseVector([-0.0001, 0.0034, 0.0009, 0.0041, 0.0053]),
 DenseVector([0.0007, 0.0023, 0.0004, 0.004, 0.0033]),
 DenseVector([-0.0, 0.0018, 0.0004, 0.0041, 0.0034]),
 DenseVector([-0.0001, 0.0039, 0.001, 0.0042, 0.006]),
 DenseVector([-0.002, 0.0059, 0.0019, -0.0012, 0.0025]),
 DenseVector([-0.0002, 0.005, 0.0014, 0.0007, 0.0026]),
 DenseVector([-0.0001, 0.0057, 0.0015, 0.0013, 0.0039]),
 DenseVector([0.0003, 0.0029, 0.0007, 0.0035, 0.0037]),
 DenseVector([-0.001, 0.0008, 0.0003, 0.0039, 0.0028]),
 DenseVector([-0.0, 0.0031, 0.0008, 0.0032, 0.0037]),
 DenseVector([0.0003, 0.0021, 0.0004, 0.0049, 0.0047]),
 DenseVector([0.0003, 0.002, 0.0004, 0.007, 0.0074]),
 DenseVector([-0.0004, 0.0097, 0.0027, -0.0025, 0.0038]),
 DenseVector([-0.0005, 0.0018, 0.0005, 0.0039, 0.0035]),
 DenseVector([-0.0, 0.0048, 0.0013, 0.0006, 0.0022]),
 DenseVector([0.0003, 0.0017, 0.0003, 0.007, 0.007]),
 DenseVector([-0.0006, 0.0202, 0.0057, -0.0112, 0.0042]),
 DenseVector([-0.0005, 0.0028, 0.0008, 0.004, 0.0048]),
 DenseVector([-0.0008, 0.0019, 0.0006, 0.0051, 0.0056]),
 DenseVector([-0.0006, 0.0015, 0.0004, 0.0056, 0.0056]),
 DenseVector([-0.0003, 0.0106, 0.0029, -0.0031, 0.004]),
 DenseVector([-0.0008, 0.008, 0.0023, -0.0009, 0.0044]),
 DenseVector([-0.0006, 0.0033, 0.0009, 0.0022, 0.0029]),
 DenseVector([-0.0002, 0.0014, 0.0004, 0.0039, 0.0029]),
 DenseVector([-0.0003, 0.0047, 0.0012, 0.004, 0.0068]),
 DenseVector([-0.0001, 0.0048, 0.0013, 0.0017, 0.0037]),
 DenseVector([-0.0018, 0.0008, 0.0004, 0.0045, 0.0042]),
 DenseVector([-0.001, 0.0041, 0.0012, 0.0018, 0.0036]),
 DenseVector([-0.0002, 0.0031, 0.0008, 0.0028, 0.0033]),
 DenseVector([-0.0052, 0.0005, 0.0009, 0.004, 0.0059]),
 DenseVector([-0.0, 0.0024, 0.0006, 0.0021, 0.0015]),
 DenseVector([-0.001, 0.0022, 0.0007, 0.0035, 0.0039]),
 DenseVector([-0.0004, 0.003, 0.0008, 0.0053, 0.0066]),
 DenseVector([-0.0005, 0.0013, 0.0004, 0.0038, 0.0029]),
 DenseVector([-0.0004, 0.0065, 0.0018, -0.0007, 0.0025]),
 DenseVector([-0.0, 0.004, 0.0011, 0.0017, 0.0027]),
 DenseVector([-0.0002, 0.0011, 0.0003, 0.0045, 0.0032]),
 DenseVector([-0.0008, 0.003, 0.0008, 0.0054, 0.0071]),
 DenseVector([-0.0007, 0.0025, 0.0008, 0.0025, 0.0027]),
 DenseVector([-0.0006, 0.0016, 0.0005, 0.0042, 0.0037]),
 DenseVector([-0.0003, 0.0152, 0.0042, -0.0058, 0.0055]),
 DenseVector([-0.0005, 0.0015, 0.0004, 0.0033, 0.0025]),
 DenseVector([-0.0, 0.0017, 0.0004, 0.0045, 0.0039]),
 DenseVector([-0.0001, 0.0039, 0.001, 0.0024, 0.0037]),
 DenseVector([-0.0004, 0.0038, 0.001, 0.0025, 0.0039]),
 DenseVector([-0.0001, 0.0014, 0.0004, 0.004, 0.0029]),
 DenseVector([-0.0001, 0.0019, 0.0005, 0.0043, 0.0038]),
 DenseVector([-0.0001, 0.0027, 0.0007, 0.0034, 0.0037]),
 DenseVector([-0.0, 0.0031, 0.0008, 0.002, 0.0021]),
 DenseVector([-0.0004, 0.0033, 0.0009, 0.0035, 0.0047]),
 DenseVector([-0.0004, 0.0024, 0.0006, 0.004, 0.0042]),
 DenseVector([-0.0003, 0.0025, 0.0007, 0.0028, 0.0027]),
 DenseVector([-0.0006, 0.0015, 0.0004, 0.0056, 0.0056]),
 DenseVector([-0.0012, 0.0027, 0.0009, 0.0017, 0.0021]),
 DenseVector([0.0004, 0.0081, 0.0021, 0.0016, 0.0069]),
 DenseVector([-0.0004, 0.0038, 0.0011, 0.0015, 0.0025]),
 DenseVector([-0.0003, 0.0023, 0.0005, 0.0068, 0.008]),
 DenseVector([-0.0001, 0.0045, 0.0012, 0.0009, 0.0022]),
 DenseVector([-0.0003, 0.0002, -0.0, 0.0085, 0.0079]),
 DenseVector([-0.0002, 0.002, 0.0005, 0.005, 0.005]),
 DenseVector([-0.0002, 0.0105, 0.0029, -0.0029, 0.004]),
 DenseVector([-0.0005, 0.0012, 0.0003, 0.0062, 0.006]),
 DenseVector([-0.0002, 0.0032, 0.0008, 0.0038, 0.0049]),
 DenseVector([-0.0008, 0.0033, 0.0009, 0.0048, 0.0066]),
 DenseVector([-0.003, 0.0013, 0.0008, 0.0034, 0.0042]),
 DenseVector([-0.0, 0.0034, 0.0009, 0.0022, 0.0027]),
 DenseVector([-0.0003, 0.0006, 0.0001, 0.0066, 0.0056]),
 DenseVector([-0.0006, 0.006, 0.0017, -0.0002, 0.003]),
 DenseVector([-0.0004, 0.0025, 0.0007, 0.0032, 0.0033]),
 DenseVector([-0.0, 0.0052, 0.0014, 0.0001, 0.0016]),
 DenseVector([-0.0005, 0.0028, 0.0008, 0.005, 0.0062]),
 DenseVector([-0.0107, -0.0018, 0.0013, 0.002, 0.0046]),
 DenseVector([-0.0001, 0.0024, 0.0006, 0.0047, 0.0047]),
 DenseVector([-0.0001, 0.0035, 0.0008, 0.0049, 0.0062]),
 DenseVector([-0.0001, 0.0027, 0.0007, 0.0028, 0.0027]),
 DenseVector([-0.0001, 0.0014, 0.0003, 0.006, 0.0056]),
 DenseVector([-0.0009, 0.0035, 0.0011, 0.0019, 0.0031]),
 DenseVector([-0.0014, 0.0056, 0.0017, 0.0032, 0.0076]),
 DenseVector([-0.0002, 0.0017, 0.0004, 0.0051, 0.0048]),
 DenseVector([-0.0124, 0.001, 0.0024, -0.002, 0.004]),
 DenseVector([-0.0001, 0.0009, 0.0002, 0.0036, 0.0019]),
 DenseVector([-0.0, 0.003, 0.0008, 0.0034, 0.0039]),
 DenseVector([-0.0001, 0.002, 0.0005, 0.0041, 0.0037]),
 DenseVector([-0.0825, -0.0249, 0.0075, -0.0211, 0.003]),
 DenseVector([-0.0003, 0.0039, 0.001, 0.0029, 0.0044]),
 DenseVector([-0.0006, 0.0052, 0.0015, 0.0011, 0.0037]),
 DenseVector([-0.0008, 0.0056, 0.0016, -0.0003, 0.0023]),
 DenseVector([0.0003, 0.0053, 0.0014, 0.0008, 0.0028]),
 DenseVector([-0.0002, 0.0033, 0.0009, 0.0019, 0.0023]),
 DenseVector([-0.0001, 0.0019, 0.0005, 0.0039, 0.0033]),
 DenseVector([-0.0018, 0.0025, 0.0009, 0.0053, 0.0071]),
 DenseVector([0.0001, 0.0058, 0.0015, 0.0032, 0.0067]),
 DenseVector([-0.0001, 0.0031, 0.0008, 0.0026, 0.0029]),
 DenseVector([0.0001, 0.0086, 0.0022, 0.0014, 0.0073]),
 DenseVector([-0.0, 0.0015, 0.0003, 0.0039, 0.0029]),
 DenseVector([0.0001, 0.0031, 0.0008, 0.0037, 0.0043]),
 DenseVector([0.0005, 0.0011, 0.0001, 0.0054, 0.0041]),
 DenseVector([0.0013, 0.0112, 0.0028, -0.0003, 0.0072]),
 DenseVector([-0.0, 0.0015, 0.0003, 0.0048, 0.0041]),
 DenseVector([-0.0001, 0.0017, 0.0004, 0.006, 0.006]),
 DenseVector([-0.0, 0.0016, 0.0003, 0.0065, 0.0066]),
 DenseVector([-0.0002, 0.0009, 0.0002, 0.0035, 0.0016]),
 DenseVector([-0.0, 0.0024, 0.0006, 0.0022, 0.0015]),
 DenseVector([-0.0001, 0.0015, 0.0003, 0.0049, 0.0042]),
 DenseVector([-0.0, 0.006, 0.0016, 0.0022, 0.0057]),
 DenseVector([0.0001, 0.0033, 0.0009, 0.0022, 0.0025]),
 DenseVector([-0.0002, 0.0034, 0.0009, 0.0051, 0.0067]),
 DenseVector([-0.0008, 0.0028, 0.0008, 0.0022, 0.0024]),
 DenseVector([-0.0, 0.0076, 0.0021, -0.0011, 0.003]),
 DenseVector([-0.0008, 0.0035, 0.001, 0.0022, 0.0033]),
 DenseVector([-0.0021, 0.0, 0.0003, 0.0059, 0.0053]),
 DenseVector([-0.0029, 0.0042, 0.0016, -0.0002, 0.0025]),
 DenseVector([-0.0026, 0.001, 0.0007, 0.0031, 0.0032]),
 DenseVector([-0.0002, 0.005, 0.0014, 0.0003, 0.0021]),
 DenseVector([-0.0003, 0.0056, 0.0015, 0.003, 0.0064]),
 DenseVector([-0.0003, 0.0016, 0.0004, 0.0037, 0.0028]),
 DenseVector([-0.0002, 0.0005, 0.0001, 0.0053, 0.0036]),
 DenseVector([-0.0001, 0.0044, 0.0011, 0.0029, 0.0049]),
 DenseVector([-0.0009, 0.0, 0.0001, 0.0059, 0.0046]),
 DenseVector([-0.0004, 0.0115, 0.0032, -0.0053, 0.002]),
 DenseVector([-0.0007, 0.0029, 0.0008, 0.0059, 0.0075]),
 DenseVector([0.0, 0.0021, 0.0005, 0.0042, 0.0038]),
 DenseVector([-0.0002, 0.0007, 0.0001, 0.0053, 0.004]),
 DenseVector([-0.0005, 0.021, 0.0059, -0.0121, 0.0038]),
 DenseVector([-0.0013, 0.0012, 0.0004, 0.0069, 0.0075]),
 DenseVector([-0.0003, 0.0024, 0.0006, 0.0046, 0.005]),
 DenseVector([-0.0012, 0.0024, 0.0008, 0.0047, 0.0058]),
 DenseVector([-0.0002, 0.0037, 0.001, 0.0038, 0.0055]),
 DenseVector([-0.0001, 0.0024, 0.0006, 0.0046, 0.0047]),
 DenseVector([-0.0021, 0.0023, 0.0009, 0.0044, 0.0059]),
 DenseVector([-0.001, 0.0014, 0.0005, 0.0029, 0.0019]),
 DenseVector([-0.0011, 0.0068, 0.002, 0.0018, 0.0068]),
 DenseVector([-0.0, 0.0014, 0.0003, 0.0036, 0.0023]),
 DenseVector([-0.0004, 0.0007, 0.0002, 0.0055, 0.0044]),
 DenseVector([-0.0, 0.0058, 0.0016, 0.0004, 0.003]),
 DenseVector([-0.0008, 0.0013, 0.0004, 0.0056, 0.0056]),
 DenseVector([-0.0004, 0.0068, 0.0018, 0.0029, 0.0078]),
 DenseVector([0.0012, 0.0028, 0.0005, 0.0033, 0.0025]),
 DenseVector([-0.0002, 0.0016, 0.0004, 0.0044, 0.0038]),
 DenseVector([-0.0001, 0.0052, 0.0013, 0.0048, 0.0082]),
 DenseVector([-0.0001, 0.0035, 0.0009, 0.0022, 0.0029]),
 DenseVector([-0.0024, 0.0045, 0.0016, 0.0001, 0.0029]),
 DenseVector([-0.0004, 0.0039, 0.001, 0.0045, 0.0066]),
 DenseVector([-0.0011, 0.002, 0.0006, 0.0042, 0.0046]),
 DenseVector([-0.0002, 0.0082, 0.0023, -0.0021, 0.0024]),
 DenseVector([-0.0003, 0.0015, 0.0004, 0.0038, 0.0028]),
 DenseVector([-0.0003, 0.0064, 0.0017, 0.0016, 0.0054]),
 DenseVector([-0.0021, 0.0048, 0.0016, 0.0026, 0.0064]),
 DenseVector([-0.0005, 0.0037, 0.001, 0.0029, 0.0043]),
 DenseVector([-0.0002, 0.0029, 0.0007, 0.0051, 0.0062]),
 DenseVector([-0.0003, 0.0087, 0.0024, -0.0008, 0.0048]),
 DenseVector([-0.0, 0.0008, 0.0001, 0.0053, 0.0038]),
 DenseVector([-0.0001, 0.005, 0.0013, 0.0015, 0.0035]),
 DenseVector([-0.0012, 0.0019, 0.0006, 0.0036, 0.0037]),
 DenseVector([-0.0006, 0.0005, 0.0002, 0.0044, 0.0029]),
 DenseVector([-0.0, 0.001, 0.0002, 0.005, 0.0038]),
 DenseVector([-0.0004, 0.0007, 0.0001, 0.0071, 0.0066]),
 DenseVector([-0.0004, 0.0018, 0.0005, 0.0028, 0.0021]),
 DenseVector([-0.0005, 0.0062, 0.0017, 0.0003, 0.0037]),
 DenseVector([-0.0007, 0.0073, 0.0021, -0.0011, 0.0032]),
 DenseVector([-0.0001, 0.0035, 0.0009, 0.0036, 0.0049]),
 DenseVector([-0.002, 0.0141, 0.0042, -0.0054, 0.0062]),
 DenseVector([-0.0001, 0.0032, 0.0008, 0.0025, 0.0029]),
 ...]

~~~

~~~python
s

~~~
Output:
~~~
DenseVector([709656.9403, 74266.7144, 22052.8227, 6458.1862, 1846.0544])

~~~

~~~python
V

~~~
Output:
~~~
DenseMatrix(16, 5, [-0.0053, -0.9994, -0.002, -0.0335, -0.0003, -0.005, -0.0001, -0.0003, ..., -0.0064, 0.0045, -0.0003, 0.0016, -0.0013, 0.0028, 0.0086, 0.0011], 0)

~~~

Chi-Square selector

~~~python
from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.linalg import Vectors

selector = ChiSqSelector(numTopFeatures=6, featuresCol="features",
                         outputCol="selectedFeatures", labelCol="y")

chi_selector = selector.fit(df)
    
result = chi_selector.transform(df)

print("ChiSqSelector output with top %d features selected" % selector.getNumTopFeatures())
print("Selected Indices: ", chi_selector.selectedFeatures)
result.show()

~~~
Output:
~~~
ChiSqSelector output with top 6 features selected
Selected Indices:  [0, 1, 2, 3, 4, 5]
+---+--------------------+--------------------+---+-------+---+--------+--------+-----+--------+----+-------+---------+-------+-------+----+-------+-----+--------+--------------------+
|  y|           features2|            features|age|balance|day|duration|campaign|pdays|previous| job|marital|education|default|housing|loan|contact|month|poutcome|    selectedFeatures|
+---+--------------------+--------------------+---+-------+---+--------+--------+-----+--------+----+-------+---------+-------+-------+----+-------+-----+--------+--------------------+
|0.0|(32,[0,1,2,3,4,5,...|(16,[0,1,2,3,4,5,...| 58|   2143|  5|     261|       1|   -1|       0| 1.0|    0.0|      1.0|    0.0|    0.0| 0.0|    1.0|  0.0|     0.0|(6,[0,1,2,3,4,5],...|
|0.0|(32,[0,1,2,3,4,5,...|(16,[0,1,2,3,4,5,...| 44|     29|  5|     151|       1|   -1|       0| 2.0|    1.0|      0.0|    0.0|    0.0| 0.0|    1.0|  0.0|     0.0|(6,[0,1,2,3,4,5],...|
|0.0|(32,[0,1,2,3,4,5,...|(16,[0,1,2,3,4,5,...| 33|      2|  5|      76|       1|   -1|       0| 7.0|    0.0|      0.0|    0.0|    0.0| 1.0|    1.0|  0.0|     0.0|(6,[0,1,2,3,4,5],...|
|0.0|(32,[0,1,2,3,4,5,...|(16,[0,1,2,3,4,5,...| 47|   1506|  5|      92|       1|   -1|       0| 0.0|    0.0|      3.0|    0.0|    0.0| 0.0|    1.0|  0.0|     0.0|(6,[0,1,2,3,4,5],...|
|0.0|[3.10770689395434...|[33.0,1.0,5.0,198...| 33|      1|  5|     198|       1|   -1|       0|11.0|    1.0|      3.0|    0.0|    1.0| 0.0|    1.0|  0.0|     0.0|[33.0,1.0,5.0,198...|
|0.0|(32,[0,1,2,3,4,5,...|(16,[0,1,2,3,4,5,...| 35|    231|  5|     139|       1|   -1|       0| 1.0|    0.0|      1.0|    0.0|    0.0| 0.0|    1.0|  0.0|     0.0|(6,[0,1,2,3,4,5],...|
|0.0|[2.63684221305217...|[28.0,447.0,5.0,2...| 28|    447|  5|     217|       1|   -1|       0| 1.0|    1.0|      1.0|    0.0|    0.0| 1.0|    1.0|  0.0|     0.0|[28.0,447.0,5.0,2...|
|0.0|[3.95526331957825...|[42.0,2.0,5.0,380...| 42|      2|  5|     380|       1|   -1|       0| 7.0|    2.0|      1.0|    1.0|    0.0| 0.0|    1.0|  0.0|     0.0|[42.0,2.0,5.0,380...|
|0.0|(32,[0,1,2,3,4,5,...|(16,[0,1,2,3,4,5,...| 58|    121|  5|      50|       1|   -1|       0| 5.0|    0.0|      2.0|    0.0|    0.0| 0.0|    1.0|  0.0|     0.0|(6,[0,1,2,3,4,5],...|
|0.0|(32,[0,1,2,3,4,5,...|(16,[0,1,2,3,4,5,...| 43|    593|  5|      55|       1|   -1|       0| 2.0|    1.0|      0.0|    0.0|    0.0| 0.0|    1.0|  0.0|     0.0|(6,[0,1,2,3,4,5],...|
|0.0|(32,[0,1,2,3,4,5,...|(16,[0,1,2,3,4,5,...| 41|    270|  5|     222|       1|   -1|       0| 3.0|    2.0|      0.0|    0.0|    0.0| 0.0|    1.0|  0.0|     0.0|(6,[0,1,2,3,4,5],...|
|0.0|(32,[0,1,2,3,4,5,...|(16,[0,1,2,3,4,5,...| 29|    390|  5|     137|       1|   -1|       0| 3.0|    1.0|      0.0|    0.0|    0.0| 0.0|    1.0|  0.0|     0.0|(6,[0,1,2,3,4,5],...|
|0.0|(32,[0,1,2,3,4,5,...|(16,[0,1,2,3,4,5,...| 53|      6|  5|     517|       1|   -1|       0| 2.0|    0.0|      0.0|    0.0|    0.0| 0.0|    1.0|  0.0|     0.0|(6,[0,1,2,3,4,5],...|
|0.0|(32,[0,1,2,3,4,5,...|(16,[0,1,2,3,4,5,...| 58|     71|  5|      71|       1|   -1|       0| 2.0|    0.0|      3.0|    0.0|    0.0| 0.0|    1.0|  0.0|     0.0|(6,[0,1,2,3,4,5],...|
|0.0|(32,[0,1,2,3,4,5,...|(16,[0,1,2,3,4,5,...| 57|    162|  5|     174|       1|   -1|       0| 4.0|    0.0|      0.0|    0.0|    0.0| 0.0|    1.0|  0.0|     0.0|(6,[0,1,2,3,4,5],...|
|0.0|(32,[0,1,2,3,4,5,...|(16,[0,1,2,3,4,5,...| 51|    229|  5|     353|       1|   -1|       0| 5.0|    0.0|      2.0|    0.0|    0.0| 0.0|    1.0|  0.0|     0.0|(6,[0,1,2,3,4,5],...|
|0.0|(32,[0,1,2,3,4,5,...|[45.0,13.0,5.0,98...| 45|     13|  5|      98|       1|   -1|       0| 3.0|    1.0|      3.0|    0.0|    0.0| 0.0|    1.0|  0.0|     0.0|[45.0,13.0,5.0,98...|
|0.0|(32,[0,1,2,3,4,5,...|(16,[0,1,2,3,4,5,...| 57|     52|  5|      38|       1|   -1|       0| 0.0|    0.0|      2.0|    0.0|    0.0| 0.0|    1.0|  0.0|     0.0|(6,[0,1,2,3,4,5],...|
|0.0|(32,[0,1,2,3,4,5,...|(16,[0,1,2,3,4,5,...| 60|     60|  5|     219|       1|   -1|       0| 5.0|    0.0|      2.0|    0.0|    0.0| 0.0|    1.0|  0.0|     0.0|(6,[0,1,2,3,4,5],...|
|0.0|(32,[0,2,3,4,5,7,...|(16,[0,2,3,4,5,7,...| 33|      0|  5|      54|       1|   -1|       0| 4.0|    0.0|      0.0|    0.0|    0.0| 0.0|    1.0|  0.0|     0.0|(6,[0,2,3,4,5],[3...|
+---+--------------------+--------------------+---+-------+---+--------+--------+-----+--------+----+-------+---------+-------+-------+----+-------+-----+--------+--------------------+
only showing top 20 rows


~~~

~~~python
features_df['chisq_importance'] = features_df['idx'].apply(lambda x: 1 if x in chi_selector.selectedFeatures else 0)

~~~

~~~python
features_df

~~~
Output:
~~~
   idx       name  chisq_importance
0    0        age                 1
1    1    balance                 1
2    2        day                 1
3    3   duration                 1
4    4   campaign                 1
5    5      pdays                 1
6    6   previous                 0
0    7        job                 0
1    8    marital                 0
2    9  education                 0
3   10    default                 0
4   11    housing                 0
5   12       loan                 0
6   13    contact                 0
7   14      month                 0
8   15   poutcome                 0

~~~

Model-based Feature Selection

~~~python
from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(featuresCol='features', labelCol=target_variable_name)
rf_model = rf.fit(df)
rf_model.featureImportances

~~~
Output:
~~~
SparseVector(16, {0: 0.0321, 1: 0.0021, 2: 0.0061, 3: 0.453, 4: 0.0031, 5: 0.0129, 6: 0.0437, 7: 0.0165, 8: 0.0031, 9: 0.0049, 10: 0.0, 11: 0.0241, 12: 0.0025, 13: 0.022, 14: 0.1248, 15: 0.2492})

~~~

~~~python
#temporary output rf_output
rf_output = rf_model.featureImportances
features_df['Importance'] = features_df['idx'].apply(lambda x: rf_output[x] if x in rf_output.indices else 0)

~~~

~~~python
#sort values based on descending importance feature
features_df.sort_values("Importance", ascending=False, inplace=True)

~~~

~~~python
features_df

~~~
Output:
~~~
   idx       name  chisq_importance  Importance
3    3   duration                 1    0.453019
8   15   poutcome                 0    0.249233
7   14      month                 0    0.124839
6    6   previous                 0    0.043690
0    0        age                 1    0.032062
4   11    housing                 0    0.024059
6   13    contact                 0    0.021951
0    7        job                 0    0.016516
5    5      pdays                 1    0.012909
2    2        day                 1    0.006084
2    9  education                 0    0.004885
4    4   campaign                 1    0.003054
1    8    marital                 0    0.003052
5   12       loan                 0    0.002532
1    1    balance                 1    0.002114
3   10    default                 0    0.000002

~~~

~~~python
import matplotlib.pyplot as plt

features_df.sort_values("Importance", ascending=True, inplace=True)
plt.barh(features_df['name'], features_df['Importance'])
plt.title("Feature Importane Plot")
plt.xlabel("Importance Score")
plt.ylabel("Variable Importance")

~~~
Output:
~~~
Text(0, 0.5, 'Variable Importance')<Figure size 432x288 with 1 Axes>

~~~

# Custom-built Variable Selection Process

Information Value using weight of evidence

Correlation

~~~python
from pyspark.mllib.stat import Statistics

correlation_type = 'pearson' # 'pearson', 'spearman'

~~~

~~~python
features_df = None
for k, v in df.schema["features"].metadata["ml_attr"]["attrs"].items():
    if features_df is None:
      features_df = pd.DataFrame(v)
    else:
      features_df= pd.concat([features_df, pd.DataFrame(v)], axis=0)

column_names = list(features_df['name'])

~~~

~~~python
column_names

~~~
Output:
~~~
['age',
 'balance',
 'day',
 'duration',
 'campaign',
 'pdays',
 'previous',
 'job',
 'marital',
 'education',
 'default',
 'housing',
 'loan',
 'contact',
 'month',
 'poutcome']

~~~

~~~python
df_vector = df.rdd.map(lambda x: x['features'].toArray())

~~~

~~~python
matrix = Statistics.corr(df_vector, method=correlation_type)

~~~

~~~python
corr_df = pd.DataFrame(matrix, columns=column_names, index=column_names)
corr_df

~~~
Output:
~~~
                age   balance       day  ...   contact     month  poutcome
age        1.000000  0.097783 -0.009120  ...  0.122114  0.089717  0.012238
balance    0.097783  1.000000  0.004503  ...  0.002844  0.092853  0.037272
day       -0.009120  0.004503  1.000000  ... -0.006302 -0.038019 -0.072629
duration  -0.004648  0.021560 -0.030206  ... -0.029350  0.014097  0.023192
campaign   0.004760 -0.014578  0.162490  ...  0.046971 -0.093829 -0.094982
pdays     -0.023758  0.003435 -0.093044  ... -0.170654  0.130504  0.709008
previous   0.001288  0.016674 -0.051710  ... -0.091911  0.124014  0.485040
job        0.077468  0.020404 -0.010874  ...  0.011959  0.116516  0.022384
marital   -0.126351 -0.028172 -0.005217  ... -0.038869  0.010914  0.020126
education  0.167296  0.039067 -0.004675  ...  0.062967  0.028758 -0.010689
default   -0.017879 -0.066745  0.009424  ...  0.000961 -0.033731 -0.037940
housing    0.185513  0.068768  0.027982  ... -0.089783  0.270254  0.000527
loan      -0.015655 -0.084350  0.011370  ... -0.015964 -0.060078 -0.047586
contact    0.122114  0.002844 -0.006302  ...  1.000000 -0.175432 -0.169951
month      0.089717  0.092853 -0.038019  ... -0.175432  1.000000  0.234792
poutcome   0.012238  0.037272 -0.072629  ... -0.169951  0.234792  1.000000

[16 rows x 16 columns]

~~~

~~~python
final_corr_df = pd.DataFrame(corr_df.abs().unstack().sort_values(kind='quicksort')).reset_index()
final_corr_df.rename({'level_0': 'col1', 'level_1': 'col2', 0: 'correlation_value'}, axis=1, inplace=True)
final_corr_df = final_corr_df[final_corr_df['col1'] != final_corr_df['col2']]
final_corr_df

~~~
Output:
~~~
         col1      col2  correlation_value
0     housing  poutcome           0.000527
1    poutcome   housing           0.000527
2     default   contact           0.000961
3     contact   default           0.000961
4    duration  previous           0.001203
..        ...       ...                ...
235  previous     pdays           0.454820
236  poutcome  previous           0.485040
237  previous  poutcome           0.485040
238  poutcome     pdays           0.709008
239     pdays  poutcome           0.709008

[240 rows x 3 columns]

~~~

~~~python
column_names

~~~
Output:
~~~
['age',
 'balance',
 'day',
 'duration',
 'campaign',
 'pdays',
 'previous',
 'job',
 'marital',
 'education',
 'default',
 'housing',
 'loan',
 'contact',
 'month',
 'poutcome']

~~~

~~~python
import seaborn as sns

#ploting the heatmap for correlation
sns.set(rc = {'figure.figsize':(15,8)})
ax = sns.heatmap(matrix, annot=True)

~~~
Output:
~~~
<Figure size 1080x576 with 2 Axes>

~~~

~~~python
correlation_cutoff = 0.65 #custom parameter
final_corr_df[final_corr_df['correlation_value'] > correlation_cutoff]

~~~
Output:
~~~
         col1      col2  correlation_value
238  poutcome     pdays           0.709008
239     pdays  poutcome           0.709008

~~~

~~~python
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol
from pyspark.mllib.stat import Statistics

class CustomCorrelation(Transformer, HasInputCol):
    """
    A custom function to calculate the correlation between two variables.
    
    Parameters:
    -----------
    inputCol: default value (None)
        Feature column name to be used for the correlation purpose. The input column should be assembled vector.
        
    correlation_type: 'pearson' or 'spearman'
    
    correlation_cutoff: float, default value (0.7), accepted values 0 to 1
        Columns more than the specified cutoff will be displayed in the output dataframe. 
    """
    
    # Initialize parameters for the function
    def __init__(self, inputCol=None, correlation_type='pearson', correlation_cutoff=0.7):
        
        super(CustomCorrelation, self).__init__()
        
        assert inputCol, "Please provide a assembled feature column name"
        
        #self.inputCol is class parameter
        self.inputCol = inputCol         
        assert correlation_type == 'pearson' or correlation_type == 'spearman', "Please provide \
                                a valid option for correlation type. 'pearson' or 'spearman'. "
        
        #self.correlation_type is class parameter
        self.correlation_type = correlation_type
        assert 0.0 <= correlation_cutoff <= 1.0, "Provide a valid value for cutoff. Accepted range is 0 to 1" 
        
        #self.correlation_cutoff is class parameter
        self.correlation_cutoff = correlation_cutoff

    
    # Transformer function, method inside a class, '_transform' - protected parameter
    def _transform(self, df):
        features_df = None
        for k, v in df.schema["features"].metadata["ml_attr"]["attrs"].items():
            if features_df is None:
              features_df = pd.DataFrame(v)
            else:
              features_df= pd.concat([features_df, pd.DataFrame(v)], axis=0)
            
        #self.column_names is class parameter, created for future use
        self.column_names = list(features_df['name'])
        
        df_vector = df.rdd.map(lambda x: x[self.inputCol].toArray())
        matrix = Statistics.corr(df_vector, method=self.correlation_type)
        
        # apply pandas dataframe operation on the fit output
        corr_df = pd.DataFrame(matrix, columns=self.column_names, index=self.column_names)

        final_corr_df = pd.DataFrame(corr_df.abs().unstack().sort_values(kind='quicksort')).reset_index()
        final_corr_df.rename({'level_0': 'col1', 'level_1': 'col2', 0: 'correlation_value'}, axis=1, inplace=True)
        final_corr_df = final_corr_df[final_corr_df['col1'] != final_corr_df['col2']]
        
        #shortlisted dataframe based on custom cutoff
        shortlisted_corr_df = final_corr_df[final_corr_df['correlation_value'] > self.correlation_cutoff]

        return corr_df, shortlisted_corr_df

~~~

~~~python
clf = CustomCorrelation(inputCol='features')
output, shorlisted_output = clf.transform(df)

~~~

~~~python
output

~~~
Output:
~~~
                age   balance       day  ...   contact     month  poutcome
age        1.000000  0.097783 -0.009120  ...  0.122114  0.089717  0.012238
balance    0.097783  1.000000  0.004503  ...  0.002844  0.092853  0.037272
day       -0.009120  0.004503  1.000000  ... -0.006302 -0.038019 -0.072629
duration  -0.004648  0.021560 -0.030206  ... -0.029350  0.014097  0.023192
campaign   0.004760 -0.014578  0.162490  ...  0.046971 -0.093829 -0.094982
pdays     -0.023758  0.003435 -0.093044  ... -0.170654  0.130504  0.709008
previous   0.001288  0.016674 -0.051710  ... -0.091911  0.124014  0.485040
job        0.077468  0.020404 -0.010874  ...  0.011959  0.116516  0.022384
marital   -0.126351 -0.028172 -0.005217  ... -0.038869  0.010914  0.020126
education  0.167296  0.039067 -0.004675  ...  0.062967  0.028758 -0.010689
default   -0.017879 -0.066745  0.009424  ...  0.000961 -0.033731 -0.037940
housing    0.185513  0.068768  0.027982  ... -0.089783  0.270254  0.000527
loan      -0.015655 -0.084350  0.011370  ... -0.015964 -0.060078 -0.047586
contact    0.122114  0.002844 -0.006302  ...  1.000000 -0.175432 -0.169951
month      0.089717  0.092853 -0.038019  ... -0.175432  1.000000  0.234792
poutcome   0.012238  0.037272 -0.072629  ... -0.169951  0.234792  1.000000

[16 rows x 16 columns]

~~~

~~~python
shorlisted_output

~~~
Output:
~~~
         col1      col2  correlation_value
238  poutcome     pdays           0.709008
239     pdays  poutcome           0.709008

~~~

# Voting-based Selection

~~~python
# decision tree
from pyspark.ml.classification import DecisionTreeClassifier
dt = DecisionTreeClassifier(featuresCol='features', labelCol=target_variable_name)
dt_model = dt.fit(df)
dt_output = dt_model.featureImportances
features_df['Decision Tree'] = features_df['idx'].apply(lambda x: dt_output[x] if x in dt_output.indices else 0)

~~~

~~~python
# Gradient boosting
from pyspark.ml.classification import GBTClassifier
gb = GBTClassifier(featuresCol='features', labelCol=target_variable_name)
gb_model = gb.fit(df)
gb_output = gb_model.featureImportances
features_df['Gradient Boosting'] = features_df['idx'].apply(lambda x: gb_output[x] if x in gb_output.indices else 0)

~~~

~~~python
# Random forest
from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(featuresCol='features', labelCol=target_variable_name)
rf_model = rf.fit(df)
rf_output = rf_model.featureImportances
features_df['Random Forest'] = features_df['idx'].apply(lambda x: rf_output[x] if x in rf_output.indices else 0)

~~~

~~~python
features_df.drop(columns=['idx'], axis=1, inplace=True)

~~~

~~~python
# Voting-based selection
num_top_features = 7
columns = ['Decision Tree', 'Gradient Boosting', 'Random Forest']
score_table = pd.DataFrame({},[])
score_table['name'] = features_df['name']
for i in columns:
  score_table[i] = features_df['name'].isin(list(features_df.nlargest(num_top_features,i)['name'])).astype(int)

score_table['final_score'] = score_table.sum(axis=1)
score_table.sort_values('final_score', ascending=0)

~~~
Output:
~~~
        name  Decision Tree  Gradient Boosting  Random Forest  final_score
3   duration              1                  1              1            3
4    housing              1                  1              1            3
6    contact              1                  1              1            3
7      month              1                  1              1            3
8   poutcome              1                  1              1            3
2        day              1                  1              0            2
0        age              0                  0              1            1
1    balance              1                  0              0            1
5      pdays              0                  1              0            1
6   previous              0                  0              1            1
4   campaign              0                  0              0            0
0        job              0                  0              0            0
1    marital              0                  0              0            0
2  education              0                  0              0            0
3    default              0                  0              0            0
5       loan              0                  0              0            0

~~~

# Pipeline Compatiability for custom Transformers

~~~python
# features 컬럼을 삭제.
df = df.drop('features')

~~~

~~~python
#exclude target variable and select all other feature vectors
features_list = df.columns
features_list.remove(target_variable_name)

~~~

~~~python
# from pyspark.ml.feature import VectorAssembler
# from pyspark.ml import Pipeline

stages = []

# assemble vectors
assembler = VectorAssembler(inputCols=features_list, outputCol='features')
custom_corr = CustomCorrelation(inputCol=assembler.getOutputCol())
stages = [assembler, custom_corr]

~~~

~~~python
#use pipeline to process sequentially
pipeline = Pipeline(stages=stages)

~~~

~~~python
#pipeline model
pipelineModel = pipeline.fit(df)

~~~

~~~python
#apply pipeline model on data
output, shorlisted_output = pipelineModel.transform(df)

~~~

~~~python
shorlisted_output

~~~
Output:
~~~
              col1          col2  correlation_value
2142  features2_15         pdays           0.709008
2143         pdays  features2_15           0.709008
2144  features2_31         pdays           0.709008
2145         pdays  features2_31           0.709008
2146   features2_5  features2_31           0.709008
...            ...           ...                ...
2299      poutcome  features2_15           1.000000
2300  features2_27       housing           1.000000
2301       housing  features2_11           1.000000
2302  features2_11       housing           1.000000
2303       housing  features2_27           1.000000

[114 rows x 3 columns]

~~~
