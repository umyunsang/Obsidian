# BDA_Hands_on_Numerical_and_Textual_Data_Analytics_using_Youtube_API

# Youtube API와 함께하는 핸즈온 튜토리얼
Youtube는 No1인 비디오 공유 플랫폼이다. 1억명 사용자들은 수천억시간의 비디오를 소비하고 매분마다 500시간의 컨텐츠가 업로드된다.

다양한 형태와 장르의 영상들이 존재한다. 주로 뮤직비디오, 강좌, 버라이어티 쇼, 드라마, 상품 리뷰 등 녹화된 방송들이 게시되기도 한다. 한편, 홈쇼핑, 게이밍 대회 와 같은 영상들은 실시간으로 스트리밍 되기도 한다. 

빅데이터 4V(Volume, Velocity, Variety, Veracity) 측면에서 Youtube 관련 데이터를 관심있어야하는 이유는 다음과 같다.
- Volume: 10억명의 사용자가 생성하고 관람하는 데이터는 엄청나게 많다.
- Velocity: 다양한 스트리밍 채널에서 사용자들은 수초내에 수백개의 메시지와 함께 커뮤니케이션 및 보기가 가능하다.
- Variety: 동영상 데이터 뿐만 아니라, 구조화된  데이터(통계치, 메타데이터)와 비구조화된 텍스트(채팅, 댓글)들을 다룰 수 있다.
- Veracity: Youtube 영상 자체가 특정 사실에 대해 불확실 정보를 포함할 수 있으며, 영상에 대한 정보가 잘못 표기될 수도 있다.

# 본 튜토리얼의 기본 목표
1. Youtube API를 이용하여 영상을 검색하거나 관련된 정보를 수집할 수 있다.
2. 수집된 정보로부터 간략한 통계자료
3. 실시간 채팅메세지 분석을 진행한다.
4. 자연어 처리 도구를 사용하여 메세지의 극성/주관성을 판단해본다
5. 자막 데이터로부터 새로운 컨텐츠를 발굴해 본다.

Copyright 2023 by datasciencelabs.org

# 사전조건
1. Youtube API를 활용하기 위해서는 Google API Python Client Library.로부터 API Key를 발급받아야 한다.
2. 개인 컴퓨터를 사용하는 경우, 가능한 Linux를 이용해서 설치(install)부분을 설치완료해야한다.

# 설치하기

Conda install

~~~python
# !pip install -q condacolab
# import condacolab
# condacolab.install()

~~~
Output:
~~~
[33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
[0m✨🍰✨ Everything looks OK!

~~~

~~~python
API_KEY = "AIzaSyCt74iOovLdzJMGCfsCAW4nAssQB8LJWo0"

~~~

install the google api python client

~~~python
# !conda install -c conda-forge google-api-python-client

~~~
Output:
~~~
Collecting package metadata (current_repodata.json): - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / done
Solving environment: \ | / - \ | / - \ | / - \ | / - done

## Package Plan ##

  environment location: /usr/local

  added / updated specs:
    - google-api-python-client


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    aiohttp-3.8.4              |   py39h72bdee0_0         432 KB  conda-forge
    aiosignal-1.3.1            |     pyhd8ed1ab_0          12 KB  conda-forge
    async-timeout-4.0.2        |     pyhd8ed1ab_0           9 KB  conda-forge
    attrs-22.2.0               |     pyh71513ae_0          53 KB  conda-forge
    boltons-23.0.0             |     pyhd8ed1ab_0         296 KB  conda-forge
    cachetools-5.3.0           |     pyhd8ed1ab_0          14 KB  conda-forge
    conda-23.3.1               |   py39hf3d152e_0         933 KB  conda-forge
    frozenlist-1.3.3           |   py39hb9d737c_0          44 KB  conda-forge
    google-api-core-2.11.0     |     pyhd8ed1ab_0          75 KB  conda-forge
    google-api-python-client-2.83.0|     pyhd8ed1ab_0         5.1 MB  conda-forge
    google-auth-2.17.1         |     pyh1a96a4e_0          97 KB  conda-forge
    google-auth-httplib2-0.1.0 |     pyhd8ed1ab_1          13 KB  conda-forge
    googleapis-common-protos-1.57.1|     pyhd8ed1ab_0         114 KB  conda-forge
    httplib2-0.22.0            |     pyhd8ed1ab_0          93 KB  conda-forge
    jsonpatch-1.32             |     pyhd8ed1ab_0          14 KB  conda-forge
    jsonpointer-2.0            |             py_0           9 KB  conda-forge
    libprotobuf-3.21.12        |       h3eb15da_0         2.1 MB  conda-forge
    multidict-6.0.4            |   py39h72bdee0_0          51 KB  conda-forge
    openssl-3.1.0              |       h0b41bf4_0         2.5 MB  conda-forge
    packaging-23.0             |     pyhd8ed1ab_0          40 KB  conda-forge
    protobuf-4.21.12           |   py39h227be39_0         315 KB  conda-forge
    pyasn1-0.4.8               |             py_0          53 KB  conda-forge
    pyasn1-modules-0.2.7       |             py_0          60 KB  conda-forge
    pyparsing-3.0.9            |     pyhd8ed1ab_0          79 KB  conda-forge
    pyu2f-0.1.5                |     pyhd8ed1ab_0          31 KB  conda-forge
    rsa-4.9                    |     pyhd8ed1ab_0          29 KB  conda-forge
    six-1.16.0                 |     pyh6c4a22f_0          14 KB  conda-forge
    typing-extensions-4.5.0    |       hd8ed1ab_0           9 KB  conda-forge
    typing_extensions-4.5.0    |     pyha770c72_0          31 KB  conda-forge
    uritemplate-4.1.1          |     pyhd8ed1ab_0          12 KB  conda-forge
    yarl-1.8.2                 |   py39hb9d737c_0          87 KB  conda-forge
    ------------------------------------------------------------
                                           Total:        12.6 MB

The following NEW packages will be INSTALLED:

  aiohttp            conda-forge/linux-64::aiohttp-3.8.4-py39h72bdee0_0 
  aiosignal          conda-forge/noarch::aiosignal-1.3.1-pyhd8ed1ab_0 
  async-timeout      conda-forge/noarch::async-timeout-4.0.2-pyhd8ed1ab_0 
  attrs              conda-forge/noarch::attrs-22.2.0-pyh71513ae_0 
  boltons            conda-forge/noarch::boltons-23.0.0-pyhd8ed1ab_0 
  cachetools         conda-forge/noarch::cachetools-5.3.0-pyhd8ed1ab_0 
  frozenlist         conda-forge/linux-64::frozenlist-1.3.3-py39hb9d737c_0 
  google-api-core    conda-forge/noarch::google-api-core-2.11.0-pyhd8ed1ab_0 
  google-api-python~ conda-forge/noarch::google-api-python-client-2.83.0-pyhd8ed1ab_0 
  google-auth        conda-forge/noarch::google-auth-2.17.1-pyh1a96a4e_0 
  google-auth-httpl~ conda-forge/noarch::google-auth-httplib2-0.1.0-pyhd8ed1ab_1 
  googleapis-common~ conda-forge/noarch::googleapis-common-protos-1.57.1-pyhd8ed1ab_0 
  httplib2           conda-forge/noarch::httplib2-0.22.0-pyhd8ed1ab_0 
  jsonpatch          conda-forge/noarch::jsonpatch-1.32-pyhd8ed1ab_0 
  jsonpointer        conda-forge/noarch::jsonpointer-2.0-py_0 
  libprotobuf        conda-forge/linux-64::libprotobuf-3.21.12-h3eb15da_0 
  multidict          conda-forge/linux-64::multidict-6.0.4-py39h72bdee0_0 
  packaging          conda-forge/noarch::packaging-23.0-pyhd8ed1ab_0 
  protobuf           conda-forge/linux-64::protobuf-4.21.12-py39h227be39_0 
  pyasn1             conda-forge/noarch::pyasn1-0.4.8-py_0 
  pyasn1-modules     conda-forge/noarch::pyasn1-modules-0.2.7-py_0 
  pyparsing          conda-forge/noarch::pyparsing-3.0.9-pyhd8ed1ab_0 
  pyu2f              conda-forge/noarch::pyu2f-0.1.5-pyhd8ed1ab_0 
  rsa                conda-forge/noarch::rsa-4.9-pyhd8ed1ab_0 
  six                conda-forge/noarch::six-1.16.0-pyh6c4a22f_0 
  typing-extensions  conda-forge/noarch::typing-extensions-4.5.0-hd8ed1ab_0 
  typing_extensions  conda-forge/noarch::typing_extensions-4.5.0-pyha770c72_0 
  uritemplate        conda-forge/noarch::uritemplate-4.1.1-pyhd8ed1ab_0 
  yarl               conda-forge/linux-64::yarl-1.8.2-py39hb9d737c_0 

The following packages will be UPDATED:

  conda                              22.11.1-py39hf3d152e_1 --> 23.3.1-py39hf3d152e_0 
  openssl                                  3.0.8-h0b41bf4_0 --> 3.1.0-h0b41bf4_0 



Downloading and Extracting Packages
six-1.16.0           | 14 KB     | :   0% 0/1 [00:00<?, ?it/s]
typing-extensions-4. | 9 KB      | :   0% 0/1 [00:00<?, ?it/s][A

multidict-6.0.4      | 51 KB     | :   0% 0/1 [00:00<?, ?it/s][A[A


async-timeout-4.0.2  | 9 KB      | :   0% 0/1 [00:00<?, ?it/s][A[A[A



aiohttp-3.8.4        | 432 KB    | :   0% 0/1 [00:00<?, ?it/s][A[A[A[A




jsonpatch-1.32       | 14 KB     | :   0% 0/1 [00:00<?, ?it/s][A[A[A[A[A





libprotobuf-3.21.12  | 2.1 MB    | :   0% 0/1 [00:00<?, ?it/s][A[A[A[A[A[A






google-auth-2.17.1   | 97 KB     | :   0% 0/1 [00:00<?, ?it/s][A[A[A[A[A[A[A







packaging-23.0       | 40 KB     | :   0% 0/1 [00:00<?, ?it/s][A[A[A[A[A[A[A[A








aiosignal-1.3.1      | 12 KB     | :   0% 0/1 [00:00<?, ?it/s][A[A[A[A[A[A[A[A[A









googleapis-common-pr | 114 KB    | :   0% 0/1 [00:00<?, ?it/s][A[A[A[A[A[A[A[A[A[A










pyu2f-0.1.5          | 31 KB     | :   0% 0/1 [00:00<?, ?it/s][A[A[A[A[A[A[A[A[A[A[A











google-auth-httplib2 | 13 KB     | :   0% 0/1 [00:00<?, ?it/s][A[A[A[A[A[A[A[A[A[A[A[A












conda-23.3.1         | 933 KB    | :   0% 0/1 [00:00<?, ?it/s][A[A[A[A[A[A[A[A[A[A[A[A[A













cachetools-5.3.0     | 14 KB     | :   0% 0/1 [00:00<?, ?it/s][A[A[A[A[A[A[A[A[A[A[A[A[A[A














httplib2-0.22.0      | 93 KB     | :   0% 0/1 [00:00<?, ?it/s][A[A[A[A[A[A[A[A[A[A[A[A[A[A[A















google-api-core-2.11 | 75 KB     | :   0% 0/1 [00:00<?, ?it/s][A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A
















rsa-4.9              | 29 KB     | :   0% 0/1 [00:00<?, ?it/s][A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A

















pyasn1-modules-0.2.7 | 60 KB     | :   0% 0/1 [00:00<?, ?it/s][A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A


















 ... (more hidden) ...[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A
typing-extensions-4. | 9 KB      | : 100% 1.0/1 [00:00<00:00,  5.40it/s][A
typing-extensions-4. | 9 KB      | : 100% 1.0/1 [00:00<00:00,  5.40it/s][A

multidict-6.0.4      | 51 KB     | :  31% 0.31100987091875476/1 [00:00<00:00,  1.43it/s][A[A




jsonpatch-1.32       | 14 KB     | : 100% 1.0/1 [00:00<00:00,  4.74it/s][A[A[A[A[A



aiohttp-3.8.4        | 432 KB    | :   4% 0.03702657443168592/1 [00:00<00:06,  6.31s/it][A[A[A[A


six-1.16.0           | 14 KB     | : 100% 1.0/1 [00:00<00:00,  3.91it/s]





libprotobuf-3.21.12  | 2.1 MB    | :   1% 0.0074184923490005815/1 [00:00<00:30, 31.04s/it][A[A[A[A[A[A






google-auth-2.17.1   | 97 KB     | :  17% 0.16517128052099925/1 [00:00<00:01,  1.39s/it][A[A[A[A[A[A[A







packaging-23.0       | 40 KB     | :  40% 0.4028918506860768/1 [00:00<00:00,  1.58it/s][A[A[A[A[A[A[A[A








aiosignal-1.3.1      | 12 KB     | : 100% 1.0/1 [00:00<00:00,  3.81it/s][A[A[A[A[A[A[A[A[A




jsonpatch-1.32       | 14 KB     | : 100% 1.0/1 [00:00<00:00,  4.74it/s][A[A[A[A[A









googleapis-common-pr | 114 KB    | :  14% 0.13996599947034352/1 [00:00<00:01,  1.94s/it][A[A[A[A[A[A[A[A[A[A










pyu2f-0.1.5          | 31 KB     | :  51% 0.5139917179068892/1 [00:00<00:00,  1.83it/s][A[A[A[A[A[A[A[A[A[A[A











google-auth-httplib2 | 13 KB     | : 100% 1.0/1 [00:00<00:00,  3.49it/s][A[A[A[A[A[A[A[A[A[A[A[A












conda-23.3.1         | 933 KB    | :   2% 0.01715562573427628/1 [00:00<00:16, 17.25s/it][A[A[A[A[A[A[A[A[A[A[A[A[A





libprotobuf-3.21.12  | 2.1 MB    | :  96% 0.9644040053700756/1 [00:00<00:00,  3.66it/s]   [A[A[A[A[A[A














httplib2-0.22.0      | 93 KB     | :  17% 0.1728051301523014/1 [00:00<00:01,  1.84s/it][A[A[A[A[A[A[A[A[A[A[A[A[A[A[A













cachetools-5.3.0     | 14 KB     | : 100% 1.0/1 [00:00<00:00,  3.09it/s][A[A[A[A[A[A[A[A[A[A[A[A[A[A















google-api-core-2.11 | 75 KB     | :  21% 0.21214553929820018/1 [00:00<00:01,  1.52s/it][A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A


async-timeout-4.0.2  | 9 KB      | : 100% 1.0/1 [00:00<00:00,  4.12it/s][A[A[A

















pyasn1-modules-0.2.7 | 60 KB     | :  27% 0.26707963159181675/1 [00:00<00:00,  1.36s/it][A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A
















rsa-4.9              | 29 KB     | :  55% 0.5486387837792586/1 [00:00<00:00,  1.50it/s][A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A


















 ... (more hidden) ...[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A

multidict-6.0.4      | 51 KB     | : 100% 1.0/1 [00:00<00:00,  2.41it/s]                [A[A

six-1.16.0           | 14 KB     | : 100% 1.0/1 [00:00<00:00,  3.91it/s]






google-auth-2.17.1   | 97 KB     | : 100% 1.0/1 [00:00<00:00,  1.77it/s]                [A[A[A[A[A[A[A






google-auth-2.17.1   | 97 KB     | : 100% 1.0/1 [00:00<00:00,  1.77it/s][A[A[A[A[A[A[A







packaging-23.0       | 40 KB     | : 100% 1.0/1 [00:00<00:00,  1.53it/s]               [A[A[A[A[A[A[A[A







packaging-23.0       | 40 KB     | : 100% 1.0/1 [00:00<00:00,  1.53it/s][A[A[A[A[A[A[A[A








aiosignal-1.3.1      | 12 KB     | : 100% 1.0/1 [00:00<00:00,  3.81it/s][A[A[A[A[A[A[A[A[A



aiohttp-3.8.4        | 432 KB    | : 100% 1.0/1 [00:00<00:00,  1.55it/s]                [A[A[A[A



aiohttp-3.8.4        | 432 KB    | : 100% 1.0/1 [00:00<00:00,  1.55it/s][A[A[A[A










pyu2f-0.1.5          | 31 KB     | : 100% 1.0/1 [00:00<00:00,  1.30it/s]               [A[A[A[A[A[A[A[A[A[A[A










pyu2f-0.1.5          | 31 KB     | : 100% 1.0/1 [00:00<00:00,  1.30it/s][A[A[A[A[A[A[A[A[A[A[A











google-auth-httplib2 | 13 KB     | : 100% 1.0/1 [00:00<00:00,  3.49it/s][A[A[A[A[A[A[A[A[A[A[A[A














httplib2-0.22.0      | 93 KB     | : 100% 1.0/1 [00:00<00:00,  1.34it/s]               [A[A[A[A[A[A[A[A[A[A[A[A[A[A[A














httplib2-0.22.0      | 93 KB     | : 100% 1.0/1 [00:00<00:00,  1.34it/s][A[A[A[A[A[A[A[A[A[A[A[A[A[A[A









googleapis-common-pr | 114 KB    | : 100% 1.0/1 [00:00<00:00,  1.28it/s]                [A[A[A[A[A[A[A[A[A[A









googleapis-common-pr | 114 KB    | : 100% 1.0/1 [00:00<00:00,  1.28it/s][A[A[A[A[A[A[A[A[A[A













cachetools-5.3.0     | 14 KB     | : 100% 1.0/1 [00:00<00:00,  3.09it/s][A[A[A[A[A[A[A[A[A[A[A[A[A[A















google-api-core-2.11 | 75 KB     | : 100% 1.0/1 [00:00<00:00,  1.17it/s]                [A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A















google-api-core-2.11 | 75 KB     | : 100% 1.0/1 [00:00<00:00,  1.17it/s][A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A





libprotobuf-3.21.12  | 2.1 MB    | : 100% 1.0/1 [00:01<00:00,  3.66it/s]               [A[A[A[A[A[A
















rsa-4.9              | 29 KB     | : 100% 1.0/1 [00:01<00:00,  1.24s/it]               [A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A
















rsa-4.9              | 29 KB     | : 100% 1.0/1 [00:01<00:00,  1.24s/it][A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A

















pyasn1-modules-0.2.7 | 60 KB     | : 100% 1.0/1 [00:01<00:00,  1.28s/it]                [A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A

















pyasn1-modules-0.2.7 | 60 KB     | : 100% 1.0/1 [00:01<00:00,  1.28s/it][A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A












conda-23.3.1         | 933 KB    | : 100% 1.0/1 [00:01<00:00,  1.22s/it]                [A[A[A[A[A[A[A[A[A[A[A[A[A












conda-23.3.1         | 933 KB    | : 100% 1.0/1 [00:01<00:00,  1.22s/it][A[A[A[A[A[A[A[A[A[A[A[A[A


















 ... (more hidden) ...[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A


















 ... (more hidden) ...[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A


















                                                                        
                                                                        [A

                                                                        [A[A


                                                                        [A[A[A



                                                                        [A[A[A[A




                                                                        [A[A[A[A[A





                                                                        [A[A[A[A[A[A






                                                                        [A[A[A[A[A[A[A







                                                                        [A[A[A[A[A[A[A[A








                                                                        [A[A[A[A[A[A[A[A[A









                                                                        [A[A[A[A[A[A[A[A[A[A










                                                                        [A[A[A[A[A[A[A[A[A[A[A











                                                                        [A[A[A[A[A[A[A[A[A[A[A[A












                                                                        [A[A[A[A[A[A[A[A[A[A[A[A[A













                                                                        [A[A[A[A[A[A[A[A[A[A[A[A[A[A














                                                                        [A[A[A[A[A[A[A[A[A[A[A[A[A[A[A















                                                                        [A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A
















                                                                        [A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A

















                                                                        [A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A
[A

[A[A


[A[A[A



[A[A[A[A




[A[A[A[A[A





[A[A[A[A[A[A






[A[A[A[A[A[A[A







[A[A[A[A[A[A[A[A








[A[A[A[A[A[A[A[A[A









[A[A[A[A[A[A[A[A[A[A










[A[A[A[A[A[A[A[A[A[A[A
Preparing transaction: | / - done
Verifying transaction: | / - done
Executing transaction: | / - \ | / - \ | / - \ | / - \ | done

~~~

# API documentation
구체적인 Youtube API[https://developers.google.com/youtube/v3] 다음 링크에 있는 문서를 참고하시기 바랍니다.

API Reference
https://developers.google.com/youtube/v3/docs

## Query Template

PYTHON API는 다음과 같이 api.(resources).(method) 형태로 구성된다.
```
# To perform list method on playlists resource
request = youtube.playlists().list(
)
# To perform list method on videos resource
request = youtube.videos().list(
)
# to perform list method on channels resource
request = youtube.channels().list(
)
```

Search vs. Video resources

Search resource: contains information about a Youtube video, channel or playlist that matches the search parameters specified in an API request

Video resource: representes a Youtube Video

Part parameter
https://developers.google.com/youtube/v3/docs/search/list#parameters

a comma-separated list of one or more search resource properties that the API response will include. Set the parameter value to snippet.

~~~python
import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors


import io
import os

~~~

~~~python
# API information

api_service_name = "youtube"
api_version = "v3"
client_secrets_file = 'client_secret.json'
scopes = ['https://www.googleapis.com/auth/youtube.readonly']

# API client
youtube = googleapiclient.discovery.build(
    api_service_name, api_version, credentials=credentials)
# 'request' variable is the only thing you must change
# depending on the resource and method you need to use
# in your query
request = youtube.search().list(
    part="id,snippet",
    type='video',
    q="Big-Bet",
    videoDuration='short',
    videoDefinition='high',
    maxResults=10
)
# Query execution
response = request.execute()
# Print the results
response

~~~

Fields option

fields 옵션은 반환되는 아이템의 구조를 설계한다

~~~python
request = youtube.search().list(
    part="id,snippet",
    type='video',
    q="Big-Bet",
    videoDuration='short',
    videoDefinition='high',
    maxResults=10,
    fields="items(id(videoId),snippet(publishedAt,channelId,channelTitle,title,description))"
)
# Query execution
response = request.execute()
# Print the results
response

~~~

Pagetoken

수많은 결과가 있을때 지속적으로 제한된 양의 결과를 가져오기 위해서는 토큰을 통해 처리해야한다. 참고로 maxResult는 50으로 제한된다.

~~~python
# fields="nextPageToken, items(id(videoId),snippet(publishedAt,channelId,channelTitle,title,description))"

~~~

#### Exercise
page token 넣어 지속적으로 다음결과를 출력해보기

~~~python


~~~

#### Exercise
응답으로부터 VideoId, 게시일, 타이틀명을 직접 접근하여 가져오기

# 유튜브 통계 데이터 분석하기
유튜브는 소셜미디어 플랫폼으로 게시된 비디오에 대해 다양한 반응들을 파악할 수 있다.
예로 snippet에는 (viewCount, likeCount, dislikeCount, favoriteCount, commentCount) 등을 파악할 수있다.
또한, duration은 contentDetails에서 파악할 수 있다.

~~~python
# user query: Busan
busan_videos_ids = youtube.search().list(
    part="id",
    type='video',
    regionCode="KR",
    order="relevance",
    q="Busan",
    maxResults=50,
    fields="items(id(videoId))"
).execute()

~~~

~~~python
busan_stat_info = []

# For loop to obtain the information of each video
for item in busan_videos_ids['items']:
    # Getting the id
    vidId = item['id']['videoId']
    # Getting stats of the video
    r = youtube.videos().list(
        part="statistics,contentDetails",
        id=vidId,
        fields="items(statistics," + \
                     "contentDetails(duration))"
    ).execute()
    # We will only consider videos which contains all properties we need.
    # If a property is missing, then it will not appear as dictionary key,
    # this is why we need a try/catch block
    # print(r)
    content_detail = r['items'][0]['contentDetails']
    statistics = r['items'][0]['statistics']
    
    duration = content_detail['duration'] if 'duration' in content_detail else ""
    views = statistics['viewCount']  if 'viewCount' in statistics else 0
    likes = statistics['likeCount']  if 'likeCount' in statistics else 0
    dislikes = statistics['dislikeCount'] if 'dislikeCount' in statistics else 0
    favorites = statistics['favoriteCount'] if 'favoriteCount' in statistics else 0
    comments = statistics['commentCount'] if 'commentCount' in statistics else 0

    # Convert object type to the corresponding datatype.  
    stat_item = { 'id' : str(vidId), 
                'duration': str(duration),
                'views': int(views), 
                'likes' : int(likes),
                'dislikes' : int(dislikes),
                'favorites' : int(favorites),
                'comments' : int(comments) }
    busan_stat_info.append(stat_item)

# end for

~~~

## dataframe conversion

~~~python
import pandas as pd

~~~

~~~python
busan_videos = pd.DataFrame(data=busan_stat_info)
busan_videos.shape

~~~
Output:
~~~
(50, 7)

~~~

~~~python
busan_videos.head(10)

~~~
Output:
~~~
            id  duration     views   likes  dislikes  favorites  comments
0  Qh3wrmSUqaI  PT10M22S   1257913   11747         0          0       652
1  1nOIVzQYbRA   PT2M58S   2263570   12805         0          0         8
2  xLD8oWRmlAE   PT1M41S  53483909  110534         0          0      4661
3  zuZsAaxcDxw  PT17M23S    101110    1748         0          0       281
4  xA0hArKS1A8  PT17M29S     51022    1124         0          0        33
5  9ElZ1f0-oiQ  PT11M17S      6349     351         0          0        44
6  8LXyuAnxb00  PT17M57S     49897     962         0          0        51
7  eUXPHStLa6w   PT8M31S     80958    5789         0          0       129
8  AI0p4T1-f88   PT10M5S    161685    3102         0          0       266
9  PLSwlEq2wPU  PT15M19S     30028     735         0          0       125

~~~

show its decriptive statistics

~~~python
busan_videos.describe()

~~~
Output:
~~~
              views          likes  dislikes  favorites    comments
count  5.000000e+01      50.000000      50.0       50.0    50.00000
mean   3.603759e+06   16683.000000       0.0        0.0   633.96000
std    1.654850e+07   54307.749643       0.0        0.0  1615.09498
min    2.157000e+03       0.000000       0.0        0.0     0.00000
25%    2.444125e+04     231.750000       0.0        0.0    17.00000
50%    6.684450e+04    1104.000000       0.0        0.0    90.00000
75%    3.227512e+05    5473.000000       0.0        0.0   328.75000
max    1.056755e+08  323951.000000       0.0        0.0  9297.00000

~~~

draw its historgram that shows the distribtuion of numerical data

- Tips: check if you can observe a specific distribution such as power-law, long-tail, and so on

~~~python
busan_videos.hist(bins=100)

~~~
Output:
~~~
array([[<Axes: title={'center': 'views'}>,
        <Axes: title={'center': 'likes'}>],
       [<Axes: title={'center': 'dislikes'}>,
        <Axes: title={'center': 'favorites'}>],
       [<Axes: title={'center': 'comments'}>, <Axes: >]], dtype=object)<Figure size 640x480 with 6 Axes>

~~~

write the dataframe to a csv file

~~~python
busan_videos.to_csv('busan_video_statistics.csv')

~~~

#### Exercise
1만건이 넘는 특정 키워드에 관한 영상들을 검색하여 이로부터 의미있는 통계치 데이터를 추출하시오. 
- nextPageToken을 이용하여 데이터를 1만건 이상 수집
- 데이터를 Panda Dataframe로 변환
- 의미있는 분석 결과에 대해 이야기할 것

# captions downloads

~~~python
from googleapiclient.errors import HttpError

~~~

~~~python
def download_captions(youtube, video_id, api_key):

    # Get the video caption tracks
    caption_tracks = youtube.captions().list(part='id', videoId=video_id).execute()
    caption_ids = [track['id'] for track in caption_tracks['items']]

    print(caption_ids)

    # Download the TTML captions
    for caption_id in caption_ids:
        try:
            caption = youtube.captions().download(
                id=caption_id,
                tfmt='ttml'
            ).execute()

            # Save the captions to a file
            with open(f'{caption_id}.ttml', 'w') as f:
                f.write(caption)

        except HttpError as error:
             print(f'An HTTP error {error.resp.status} occurred: {error.content}')

~~~

~~~python
download_captions(youtube, 'O5xeyoRL95U', API_KEY)

~~~
Output:
~~~
['AUieDabMHuli-HcCo36ri76VN71k289-x9omXb7vgfJq6VKt_2A', 'AUieDaZxtX3HAntbRQhGmW7zi8YVU-wyP1ihw3BNE3R_']
An HTTP error 401 occurred: b'{\n  "error": {\n    "code": 401,\n    "message": "API keys are not supported by this API. Expected OAuth2 access token or other authentication credentials that assert a principal. See https://cloud.google.com/docs/authentication",\n    "errors": [\n      {\n        "message": "Login Required.",\n        "domain": "global",\n        "reason": "required",\n        "location": "Authorization",\n        "locationType": "header"\n      }\n    ],\n    "status": "UNAUTHENTICATED"\n  }\n}\n'
An HTTP error 401 occurred: b'{\n  "error": {\n    "code": 401,\n    "message": "API keys are not supported by this API. Expected OAuth2 access token or other authentication credentials that assert a principal. See https://cloud.google.com/docs/authentication",\n    "errors": [\n      {\n        "message": "Login Required.",\n        "domain": "global",\n        "reason": "required",\n        "location": "Authorization",\n        "locationType": "header"\n      }\n    ],\n    "status": "UNAUTHENTICATED"\n  }\n}\n'

~~~

# live video로부터 chat messages 분석하기

## Retrieve live videos

~~~python
STREAMING_VIDEO_ID = "MFHYb1oRJKo" #@param 

~~~

~~~python
r = youtube.videos().list(
    part='liveStreamingDetails,snippet',
    id= STREAMING_VIDEO_ID,
    fields='items(liveStreamingDetails(activeLiveChatId),snippet(title,liveBroadcastContent))'
).execute()

~~~

현재 활성중인  라이브 챗 아이디 가져오기:
'activeLiveChatId'

~~~python
chatID = r['items'][0]['liveStreamingDetails']['activeLiveChatId']
chatID

~~~
Output:
~~~
'Cg0KC01GSFliMW9SSktvKicKGFVDTktrbXM3Vl9yUVBxZVNjaUN3bUM3URILTUZIWWIxb1JKS28'

~~~

채팅 메세지 가져오기

~~~python
response = youtube.liveChatMessages().list(
    liveChatId=chatID,
    part="snippet,authorDetails",
    maxResults = 1000,
    fields="nextPageToken,items(snippet(publishedAt,displayMessage),authorDetails(channelId,displayName))"
).execute()

~~~

데이터 분석을 위해 데이터 저장

~~~python
chat_messages = []

for item in response['items']:
    msg = { 'authorChannelId': item['authorDetails']['channelId'],
            'authoChannelName': item['authorDetails']['displayName'],
            'messagePublishDate': item['snippet']['publishedAt'],
            'messageContent': item['snippet']['displayMessage'] }
    chat_messages.append(msg)
# end for

~~~

~~~python
chat_messages[0]

~~~
Output:
~~~
{'authorChannelId': 'UCqoxZvPcT35HhB13Jas6-4g',
 'authoChannelName': '༺Møøค🐉༻ණ',
 'messagePublishDate': '2023-04-05T23:09:57.268961+00:00',
 'messageContent': 'thank you'}

~~~

#### Exercise
채팅메시지는 각 요청마다 2000개의 메시지를 가져온다. 5초 주기 time.sleep(5)로 요청하여 메세지를 가져와 파일 혹은 데이터프레임에 축적하는 프로그램을 구현하라.

~~~python
### your code

~~~

#### Exercise 

채팅메세지에서 가장 많이 이야기한 사람을 찾는 알고리즘을 구현하라.

~~~python
### your code

~~~

# 자연어 텍스트 처리하기

Natural Language Processing(NLP)은 언어학, 컴퓨터과학, 인공지능 학문분야들이 서로 결합하여 수행하는 학문이다. 현재 유사어(Synonyms), 오류(Errors) 등 다양한 도전문제들에 대해서 많은 연구들이 진행되고 있다.
https://monkeylearn.com/blog/natural-language-processing-challenges/
현재 BERT 등 고도화된 사전학습된 언어 모델들이 이용가능하다. 

또한, 일반적인 자연어 처리는  여러개의 컴포넌트가 구성된 파이프라인(Pipeline) 형태가 설계된다는 점을 알고 있어야한다. Tokenizer, 를 시작으로, tagger, parser, .. 등등 다양한 형태로 분석을 진행하고 추출된 정보를 doc 모델 에 저장한다. 따라서, 목적이나 요구사항에 따라 적절하게 구성해야 한다.

이번 핸즈온 과정에서는 Spacy를 사용한 상용화된 모델을 사용하며 간략하게 내용을 다루고자 한다. 자연어 처리에 대해 관심이 있는 학생은 Spacy보다는 고도화된 언어 모델을 별도로 공부할 필요가 있다. 예를 들면, BERT, GPT3 이 될 수 있다.

CUDA Version 확인

~~~python
!nvcc --version

~~~
Output:
~~~
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Sep_21_10:33:58_PDT_2022
Cuda compilation tools, release 11.8, V11.8.89
Build cuda_11.8.r11.8/compiler.31833905_0

~~~

install spacy

~~~python
!conda install -c conda-forge spacy
!conda install -c conda-forge cupy
!python -m spacy download en_core_web_sm

~~~
Output:
~~~
Collecting package metadata (current_repodata.json): - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / done
Solving environment: \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - done

## Package Plan ##

  environment location: /usr/local

  added / updated specs:
    - spacy


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    catalogue-2.0.8            |   py39hf3d152e_1          32 KB  conda-forge
    click-8.1.3                |unix_pyhd8ed1ab_2          74 KB  conda-forge
    commonmark-0.9.1           |             py_0          46 KB  conda-forge
    confection-0.0.4           |   py39hcca971b_1          61 KB  conda-forge
    cymem-2.0.7                |   py39h5a03fae_1          42 KB  conda-forge
    cython-blis-0.7.9          |   py39h2ae25f5_1         9.0 MB  conda-forge
    dataclasses-0.8            |     pyhc8e2a94_3          10 KB  conda-forge
    future-0.18.3              |     pyhd8ed1ab_0         357 KB  conda-forge
    jinja2-3.1.2               |     pyhd8ed1ab_1          99 KB  conda-forge
    langcodes-3.3.0            |     pyhd8ed1ab_0         156 KB  conda-forge
    libblas-3.9.0              |16_linux64_openblas          13 KB  conda-forge
    libcblas-3.9.0             |16_linux64_openblas          13 KB  conda-forge
    libgfortran-ng-12.2.0      |      h69a702a_19          22 KB  conda-forge
    libgfortran5-12.2.0        |      h337968e_19         1.8 MB  conda-forge
    liblapack-3.9.0            |16_linux64_openblas          13 KB  conda-forge
    libopenblas-0.3.21         |pthreads_h78a6416_3        10.1 MB  conda-forge
    markupsafe-2.1.2           |   py39h72bdee0_0          23 KB  conda-forge
    murmurhash-1.0.9           |   py39h5a03fae_1          27 KB  conda-forge
    numpy-1.24.2               |   py39h7360e5f_0         6.4 MB  conda-forge
    pathy-0.10.1               |     pyhd8ed1ab_0          42 KB  conda-forge
    preshed-3.0.8              |   py39h5a03fae_1         121 KB  conda-forge
    pydantic-1.10.7            |   py39h72bdee0_0         2.1 MB  conda-forge
    pygments-2.14.0            |     pyhd8ed1ab_0         805 KB  conda-forge
    rich-12.6.0                |     pyhd8ed1ab_0         170 KB  conda-forge
    shellingham-1.5.1          |     pyhd8ed1ab_0          14 KB  conda-forge
    smart_open-5.2.1           |     pyhd8ed1ab_0          43 KB  conda-forge
    spacy-3.5.1                |   py39h0354152_0         5.1 MB  conda-forge
    spacy-legacy-3.0.12        |     pyhd8ed1ab_0          28 KB  conda-forge
    spacy-loggers-1.0.4        |     pyhd8ed1ab_0          15 KB  conda-forge
    srsly-2.4.6                |   py39h227be39_0         550 KB  conda-forge
    thinc-8.1.9                |   py39h0354152_0         847 KB  conda-forge
    typer-0.7.0                |     pyhd8ed1ab_0          56 KB  conda-forge
    wasabi-1.1.1               |   py39hf3d152e_1          45 KB  conda-forge
    ------------------------------------------------------------
                                           Total:        38.0 MB

The following NEW packages will be INSTALLED:

  catalogue          conda-forge/linux-64::catalogue-2.0.8-py39hf3d152e_1 
  click              conda-forge/noarch::click-8.1.3-unix_pyhd8ed1ab_2 
  commonmark         conda-forge/noarch::commonmark-0.9.1-py_0 
  confection         conda-forge/linux-64::confection-0.0.4-py39hcca971b_1 
  cymem              conda-forge/linux-64::cymem-2.0.7-py39h5a03fae_1 
  cython-blis        conda-forge/linux-64::cython-blis-0.7.9-py39h2ae25f5_1 
  dataclasses        conda-forge/noarch::dataclasses-0.8-pyhc8e2a94_3 
  future             conda-forge/noarch::future-0.18.3-pyhd8ed1ab_0 
  jinja2             conda-forge/noarch::jinja2-3.1.2-pyhd8ed1ab_1 
  langcodes          conda-forge/noarch::langcodes-3.3.0-pyhd8ed1ab_0 
  libblas            conda-forge/linux-64::libblas-3.9.0-16_linux64_openblas 
  libcblas           conda-forge/linux-64::libcblas-3.9.0-16_linux64_openblas 
  libgfortran-ng     conda-forge/linux-64::libgfortran-ng-12.2.0-h69a702a_19 
  libgfortran5       conda-forge/linux-64::libgfortran5-12.2.0-h337968e_19 
  liblapack          conda-forge/linux-64::liblapack-3.9.0-16_linux64_openblas 
  libopenblas        conda-forge/linux-64::libopenblas-0.3.21-pthreads_h78a6416_3 
  markupsafe         conda-forge/linux-64::markupsafe-2.1.2-py39h72bdee0_0 
  murmurhash         conda-forge/linux-64::murmurhash-1.0.9-py39h5a03fae_1 
  numpy              conda-forge/linux-64::numpy-1.24.2-py39h7360e5f_0 
  pathy              conda-forge/noarch::pathy-0.10.1-pyhd8ed1ab_0 
  preshed            conda-forge/linux-64::preshed-3.0.8-py39h5a03fae_1 
  pydantic           conda-forge/linux-64::pydantic-1.10.7-py39h72bdee0_0 
  pygments           conda-forge/noarch::pygments-2.14.0-pyhd8ed1ab_0 
  rich               conda-forge/noarch::rich-12.6.0-pyhd8ed1ab_0 
  shellingham        conda-forge/noarch::shellingham-1.5.1-pyhd8ed1ab_0 
  smart_open         conda-forge/noarch::smart_open-5.2.1-pyhd8ed1ab_0 
  spacy              conda-forge/linux-64::spacy-3.5.1-py39h0354152_0 
  spacy-legacy       conda-forge/noarch::spacy-legacy-3.0.12-pyhd8ed1ab_0 
  spacy-loggers      conda-forge/noarch::spacy-loggers-1.0.4-pyhd8ed1ab_0 
  srsly              conda-forge/linux-64::srsly-2.4.6-py39h227be39_0 
  thinc              conda-forge/linux-64::thinc-8.1.9-py39h0354152_0 
  typer              conda-forge/noarch::typer-0.7.0-pyhd8ed1ab_0 
  wasabi             conda-forge/linux-64::wasabi-1.1.1-py39hf3d152e_1 



Downloading and Extracting Packages
confection-0.0.4     | 61 KB     | :   0% 0/1 [00:00<?, ?it/s]
shellingham-1.5.1    | 14 KB     | :   0% 0/1 [00:00<?, ?it/s][A

jinja2-3.1.2         | 99 KB     | :   0% 0/1 [00:00<?, ?it/s][A[A


srsly-2.4.6          | 550 KB    | :   0% 0/1 [00:00<?, ?it/s][A[A[A



typer-0.7.0          | 56 KB     | :   0% 0/1 [00:00<?, ?it/s][A[A[A[A




smart_open-5.2.1     | 43 KB     | :   0% 0/1 [00:00<?, ?it/s][A[A[A[A[A





catalogue-2.0.8      | 32 KB     | :   0% 0/1 [00:00<?, ?it/s][A[A[A[A[A[A






pydantic-1.10.7      | 2.1 MB    | :   0% 0/1 [00:00<?, ?it/s][A[A[A[A[A[A[A







thinc-8.1.9          | 847 KB    | :   0% 0/1 [00:00<?, ?it/s][A[A[A[A[A[A[A[A








libgfortran-ng-12.2. | 22 KB     | :   0% 0/1 [00:00<?, ?it/s][A[A[A[A[A[A[A[A[A









liblapack-3.9.0      | 13 KB     | :   0% 0/1 [00:00<?, ?it/s][A[A[A[A[A[A[A[A[A[A










libcblas-3.9.0       | 13 KB     | :   0% 0/1 [00:00<?, ?it/s][A[A[A[A[A[A[A[A[A[A[A











spacy-3.5.1          | 5.1 MB    | :   0% 0/1 [00:00<?, ?it/s][A[A[A[A[A[A[A[A[A[A[A[A












libopenblas-0.3.21   | 10.1 MB   | :   0% 0/1 [00:00<?, ?it/s][A[A[A[A[A[A[A[A[A[A[A[A[A













spacy-loggers-1.0.4  | 15 KB     | :   0% 0/1 [00:00<?, ?it/s][A[A[A[A[A[A[A[A[A[A[A[A[A[A














pathy-0.10.1         | 42 KB     | :   0% 0/1 [00:00<?, ?it/s][A[A[A[A[A[A[A[A[A[A[A[A[A[A[A















future-0.18.3        | 357 KB    | :   0% 0/1 [00:00<?, ?it/s][A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A
















numpy-1.24.2         | 6.4 MB    | :   0% 0/1 [00:00<?, ?it/s][A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A

















preshed-3.0.8        | 121 KB    | :   0% 0/1 [00:00<?, ?it/s][A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A


















 ... (more hidden) ...[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A

jinja2-3.1.2         | 99 KB     | :  16% 0.16150941908263755/1 [00:00<00:00,  1.14it/s][A[A



typer-0.7.0          | 56 KB     | :  28% 0.2849094007581818/1 [00:00<00:00,  1.94it/s][A[A[A[A
confection-0.0.4     | 61 KB     | :  26% 0.26085018309186436/1 [00:00<00:00,  1.36it/s]




smart_open-5.2.1     | 43 KB     | :  37% 0.3730503882146679/1 [00:00<00:00,  1.90it/s][A[A[A[A[A


srsly-2.4.6          | 550 KB    | :   3% 0.029078259634924438/1 [00:00<00:06,  7.18s/it][A[A[A






pydantic-1.10.7      | 2.1 MB    | :   1% 0.007570237470919688/1 [00:00<00:27, 27.88s/it][A[A[A[A[A[A[A





catalogue-2.0.8      | 32 KB     | :  50% 0.49666545410452284/1 [00:00<00:00,  2.09it/s][A[A[A[A[A[A








libgfortran-ng-12.2. | 22 KB     | :  72% 0.7159587484705471/1 [00:00<00:00,  2.89it/s][A[A[A[A[A[A[A[A[A









liblapack-3.9.0      | 13 KB     | : 100% 1.0/1 [00:00<00:00,  3.61it/s][A[A[A[A[A[A[A[A[A[A










libcblas-3.9.0       | 13 KB     | : 100% 1.0/1 [00:00<00:00,  3.53it/s][A[A[A[A[A[A[A[A[A[A[A


srsly-2.4.6          | 550 KB    | :  52% 0.5234086734286398/1 [00:00<00:00,  2.08it/s]  [A[A[A

jinja2-3.1.2         | 99 KB     | : 100% 1.0/1 [00:00<00:00,  3.49it/s]                [A[A

jinja2-3.1.2         | 99 KB     | : 100% 1.0/1 [00:00<00:00,  3.49it/s][A[A












libopenblas-0.3.21   | 10.1 MB   | :   0% 0.0015489208557598652/1 [00:00<03:26, 206.59s/it][A[A[A[A[A[A[A[A[A[A[A[A[A
shellingham-1.5.1    | 14 KB     | : 100% 1.0/1 [00:00<00:00,  5.51it/s][A











spacy-3.5.1          | 5.1 MB    | :   0% 0.003073866413262803/1 [00:00<01:53, 113.38s/it][A[A[A[A[A[A[A[A[A[A[A[A














pathy-0.10.1         | 42 KB     | :  38% 0.3812892715848266/1 [00:00<00:00,  1.08s/it][A[A[A[A[A[A[A[A[A[A[A[A[A[A[A












libopenblas-0.3.21   | 10.1 MB   | :  20% 0.19671294868150285/1 [00:00<00:01,  1.66s/it]   [A[A[A[A[A[A[A[A[A[A[A[A[A













spacy-loggers-1.0.4  | 15 KB     | : 100% 1.0/1 [00:00<00:00,  2.36it/s][A[A[A[A[A[A[A[A[A[A[A[A[A[A











confection-0.0.4     | 61 KB     | : 100% 1.0/1 [00:00<00:00,  2.26it/s]















future-0.18.3        | 357 KB    | :   4% 0.044823812650470564/1 [00:00<00:09, 10.23s/it][A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A
















numpy-1.24.2         | 6.4 MB    | :   0% 0.002460556604816347/1 [00:00<03:06, 186.79s/it][A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A












libopenblas-0.3.21   | 10.1 MB   | :  38% 0.37638776794964723/1 [00:00<00:00,  1.04s/it][A[A[A[A[A[A[A[A[A[A[A[A[A











spacy-3.5.1          | 5.1 MB    | :  64% 0.639364213958663/1 [00:00<00:00,  1.59it/s]  [A[A[A[A[A[A[A[A[A[A[A[A
















numpy-1.24.2         | 6.4 MB    | :  37% 0.3690834907224521/1 [00:00<00:00,  1.15s/it]   [A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A

















preshed-3.0.8        | 121 KB    | :  13% 0.1326564486223453/1 [00:00<00:03,  4.38s/it][A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A







thinc-8.1.9          | 847 KB    | :   2% 0.018888349609701493/1 [00:00<00:30, 31.26s/it][A[A[A[A[A[A[A[A




smart_open-5.2.1     | 43 KB     | : 100% 1.0/1 [00:00<00:00,  1.65it/s]               [A[A[A[A[A




smart_open-5.2.1     | 43 KB     | : 100% 1.0/1 [00:00<00:00,  1.65it/s][A[A[A[A[A












libopenblas-0.3.21   | 10.1 MB   | :  52% 0.5204374075353146/1 [00:00<00:00,  1.10it/s] [A[A[A[A[A[A[A[A[A[A[A[A[A


















 ... (more hidden) ...[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A











spacy-3.5.1          | 5.1 MB    | :  92% 0.9190860575655782/1 [00:00<00:00,  1.95it/s][A[A[A[A[A[A[A[A[A[A[A[A
















numpy-1.24.2         | 6.4 MB    | :  66% 0.6569686134859647/1 [00:00<00:00,  1.38it/s][A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A





catalogue-2.0.8      | 32 KB     | : 100% 1.0/1 [00:00<00:00,  1.36it/s]                [A[A[A[A[A[A





catalogue-2.0.8      | 32 KB     | : 100% 1.0/1 [00:00<00:00,  1.36it/s][A[A[A[A[A[A







thinc-8.1.9          | 847 KB    | :   8% 0.07555339843880597/1 [00:00<00:07,  7.77s/it] [A[A[A[A[A[A[A[A












libopenblas-0.3.21   | 10.1 MB   | :  73% 0.7295417230628964/1 [00:00<00:00,  1.40it/s][A[A[A[A[A[A[A[A[A[A[A[A[A
















numpy-1.24.2         | 6.4 MB    | :  97% 0.9743804155072735/1 [00:00<00:00,  1.85it/s][A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A








libgfortran-ng-12.2. | 22 KB     | : 100% 1.0/1 [00:00<00:00,  2.89it/s]               [A[A[A[A[A[A[A[A[A












libopenblas-0.3.21   | 10.1 MB   | : 100% 0.9990539519651129/1 [00:00<00:00,  1.78it/s][A[A[A[A[A[A[A[A[A[A[A[A[A









liblapack-3.9.0      | 13 KB     | : 100% 1.0/1 [00:00<00:00,  3.61it/s][A[A[A[A[A[A[A[A[A[A







thinc-8.1.9          | 847 KB    | :  19% 0.1888834960970149/1 [00:00<00:02,  3.21s/it] [A[A[A[A[A[A[A[A










libcblas-3.9.0       | 13 KB     | : 100% 1.0/1 [00:00<00:00,  3.53it/s][A[A[A[A[A[A[A[A[A[A[A



typer-0.7.0          | 56 KB     | : 100% 1.0/1 [00:00<00:00,  1.03it/s]               [A[A[A[A



typer-0.7.0          | 56 KB     | : 100% 1.0/1 [00:00<00:00,  1.03it/s][A[A[A[A







thinc-8.1.9          | 847 KB    | :  38% 0.3777669921940298/1 [00:00<00:01,  1.63s/it][A[A[A[A[A[A[A[A







thinc-8.1.9          | 847 KB    | :  55% 0.5477621386813433/1 [00:01<00:00,  1.17s/it][A[A[A[A[A[A[A[A







thinc-8.1.9          | 847 KB    | :  79% 0.7933106836074626/1 [00:01<00:00,  1.23it/s][A[A[A[A[A[A[A[A






pydantic-1.10.7      | 2.1 MB    | : 100% 1.0/1 [00:01<00:00,  1.14s/it]                 [A[A[A[A[A[A[A






pydantic-1.10.7      | 2.1 MB    | : 100% 1.0/1 [00:01<00:00,  1.14s/it][A[A[A[A[A[A[A


srsly-2.4.6          | 550 KB    | : 100% 1.0/1 [00:01<00:00,  1.31s/it]               [A[A[A


srsly-2.4.6          | 550 KB    | : 100% 1.0/1 [00:01<00:00,  1.31s/it][A[A[A













spacy-loggers-1.0.4  | 15 KB     | : 100% 1.0/1 [00:01<00:00,  2.36it/s][A[A[A[A[A[A[A[A[A[A[A[A[A[A














pathy-0.10.1         | 42 KB     | : 100% 1.0/1 [00:01<00:00,  1.29s/it]               [A[A[A[A[A[A[A[A[A[A[A[A[A[A[A














pathy-0.10.1         | 42 KB     | : 100% 1.0/1 [00:01<00:00,  1.29s/it][A[A[A[A[A[A[A[A[A[A[A[A[A[A[A

















preshed-3.0.8        | 121 KB    | : 100% 1.0/1 [00:01<00:00,  1.26s/it]               [A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A

















preshed-3.0.8        | 121 KB    | : 100% 1.0/1 [00:01<00:00,  1.26s/it][A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A


















 ... (more hidden) ...[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A















future-0.18.3        | 357 KB    | : 100% 1.0/1 [00:01<00:00,  1.36s/it]                 [A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A















future-0.18.3        | 357 KB    | : 100% 1.0/1 [00:01<00:00,  1.36s/it][A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A
















numpy-1.24.2         | 6.4 MB    | : 100% 1.0/1 [00:03<00:00,  1.85it/s]               [A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A











spacy-3.5.1          | 5.1 MB    | : 100% 1.0/1 [00:03<00:00,  1.95it/s]               [A[A[A[A[A[A[A[A[A[A[A[A







thinc-8.1.9          | 847 KB    | : 100% 1.0/1 [00:04<00:00,  7.03s/it]               [A[A[A[A[A[A[A[A







thinc-8.1.9          | 847 KB    | : 100% 1.0/1 [00:04<00:00,  7.03s/it][A[A[A[A[A[A[A[A












libopenblas-0.3.21   | 10.1 MB   | : 100% 1.0/1 [00:06<00:00,  1.78it/s]               [A[A[A[A[A[A[A[A[A[A[A[A[A


















                                                                        
                                                                        [A

                                                                        [A[A


                                                                        [A[A[A



                                                                        [A[A[A[A




                                                                        [A[A[A[A[A





                                                                        [A[A[A[A[A[A






                                                                        [A[A[A[A[A[A[A







                                                                        [A[A[A[A[A[A[A[A








                                                                        [A[A[A[A[A[A[A[A[A









                                                                        [A[A[A[A[A[A[A[A[A[A










                                                                        [A[A[A[A[A[A[A[A[A[A[A











                                                                        [A[A[A[A[A[A[A[A[A[A[A[A












                                                                        [A[A[A[A[A[A[A[A[A[A[A[A[A













                                                                        [A[A[A[A[A[A[A[A[A[A[A[A[A[A














                                                                        [A[A[A[A[A[A[A[A[A[A[A[A[A[A[A















                                                                        [A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A
















                                                                        [A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A

















                                                                        [A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A[A
[A

[A[A


[A[A[A



[A[A[A[A




[A[A[A[A[A





[A[A[A[A[A[A






[A[A[A[A[A[A[A







[A[A[A[A[A[A[A[A








[A[A[A[A[A[A[A[A[A









[A[A[A[A[A[A[A[A[A[A










[A[A[A[A[A[A[A[A[A[A[A











[A[A[A[A[A[A[A[A[A[A[A[A












[A[A[A[A[A[A[A[A[A[A[A[A[A
Preparing transaction: | / - done
Verifying transaction: | / - \ | / done
Executing transaction: \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - done
Collecting package metadata (current_repodata.json): - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ done
Solving environment: / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | done

## Package Plan ##

  environment location: /usr/local

  added / updated specs:
    - cupy


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    cudatoolkit-11.8.0         |      h37601d7_11       635.9 MB  conda-forge
    cupy-12.0.0                |   py39hc3c280e_0        34.8 MB  conda-forge
    fastrlock-0.8              |   py39h5a03fae_3          32 KB  conda-forge
    ------------------------------------------------------------
                                           Total:       670.8 MB

The following NEW packages will be INSTALLED:

  cudatoolkit        conda-forge/linux-64::cudatoolkit-11.8.0-h37601d7_11 
  cupy               conda-forge/linux-64::cupy-12.0.0-py39hc3c280e_0 
  fastrlock          conda-forge/linux-64::fastrlock-0.8-py39h5a03fae_3 



Downloading and Extracting Packages
fastrlock-0.8        | 32 KB     | :   0% 0/1 [00:00<?, ?it/s]
cudatoolkit-11.8.0   | 635.9 MB  | :   0% 0/1 [00:00<?, ?it/s][A

cupy-12.0.0          | 34.8 MB   | :   0% 0/1 [00:00<?, ?it/s][A[A
cudatoolkit-11.8.0   | 635.9 MB  | :   0% 2.4570534161823033e-05/1 [00:00<2:47:59, 10079.54s/it][A

cupy-12.0.0          | 34.8 MB   | :   0% 0.00044855778160142137/1 [00:00<10:52, 652.51s/it][A[A
cudatoolkit-11.8.0   | 635.9 MB  | :   1% 0.005380946981439244/1 [00:00<00:50, 50.91s/it]       [A

fastrlock-0.8        | 32 KB     | :  50% 0.5007334963325183/1 [00:00<00:00,  1.13it/s]
fastrlock-0.8        | 32 KB     | : 100% 1.0/1 [00:00<00:00,  1.13it/s]               

cupy-12.0.0          | 34.8 MB   | :   4% 0.03588462252811371/1 [00:00<00:09, 10.07s/it] [A[A
cudatoolkit-11.8.0   | 635.9 MB  | :   4% 0.037814052075045644/1 [00:00<00:09, 10.27s/it][A

cupy-12.0.0          | 34.8 MB   | :   7% 0.06997501392982174/1 [00:00<00:05,  5.79s/it][A[A
cudatoolkit-11.8.0   | 635.9 MB  | :   6% 0.056045388423118334/1 [00:00<00:07,  8.03s/it][A

cupy-12.0.0          | 34.8 MB   | :  12% 0.11841925434277524/1 [00:00<00:03,  3.73s/it][A[A
cudatoolkit-11.8.0   | 635.9 MB  | :   7% 0.07034543930529934/1 [00:00<00:07,  7.69s/it] [A

cupy-12.0.0          | 34.8 MB   | :  17% 0.16955484144533728/1 [00:00<00:02,  2.93s/it][A[A
cudatoolkit-11.8.0   | 635.9 MB  | :   9% 0.08766766588938457/1 [00:00<00:06,  6.97s/it][A

cupy-12.0.0          | 34.8 MB   | :  21% 0.2099250417894652/1 [00:00<00:02,  2.79s/it] [A[A
cudatoolkit-11.8.0   | 635.9 MB  | :  10% 0.10417906484612965/1 [00:00<00:06,  6.74s/it][A

cupy-12.0.0          | 34.8 MB   | :  30% 0.30322506036256086/1 [00:01<00:01,  1.88s/it][A[A
cudatoolkit-11.8.0   | 635.9 MB  | :  12% 0.1195847897655927/1 [00:01<00:05,  6.69s/it] [A

cupy-12.0.0          | 34.8 MB   | :  38% 0.37723709432679536/1 [00:01<00:01,  1.68s/it][A[A
cudatoolkit-11.8.0   | 635.9 MB  | :  13% 0.13491680308257026/1 [00:01<00:05,  6.82s/it][A

cupy-12.0.0          | 34.8 MB   | :  44% 0.4427265304406029/1 [00:01<00:00,  1.63s/it] [A[A

cupy-12.0.0          | 34.8 MB   | :  53% 0.5266068356000687/1 [00:01<00:00,  1.47s/it][A[A

cupy-12.0.0          | 34.8 MB   | :  67% 0.6746309035285377/1 [00:01<00:00,  1.09s/it][A[A
cudatoolkit-11.8.0   | 635.9 MB  | :  15% 0.1498802583871205/1 [00:01<00:08, 10.36s/it] [A

cupy-12.0.0          | 34.8 MB   | :  82% 0.8195150669857968/1 [00:01<00:00,  1.08it/s][A[A
cudatoolkit-11.8.0   | 635.9 MB  | :  16% 0.16189524959225196/1 [00:01<00:09, 10.89s/it][A

cupy-12.0.0          | 34.8 MB   | :  99% 0.9850328883967213/1 [00:01<00:00,  1.25it/s][A[A
cudatoolkit-11.8.0   | 635.9 MB  | :  17% 0.17270628462345408/1 [00:01<00:10, 13.07s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  18% 0.1817728117291668/1 [00:01<00:11, 14.26s/it] [A
cudatoolkit-11.8.0   | 635.9 MB  | :  19% 0.18975823533175928/1 [00:02<00:12, 14.90s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  20% 0.197153966114468/1 [00:02<00:11, 14.59s/it]  [A
cudatoolkit-11.8.0   | 635.9 MB  | :  21% 0.20742444939411003/1 [00:02<00:10, 13.08s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  22% 0.21703152825138283/1 [00:02<00:09, 12.30s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  23% 0.22663860710865563/1 [00:02<00:09, 11.74s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  24% 0.23558228154355923/1 [00:02<00:11, 15.25s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  24% 0.2431254355312389/1 [00:02<00:11, 14.74s/it] [A
cudatoolkit-11.8.0   | 635.9 MB  | :  25% 0.25064401898475674/1 [00:02<00:12, 16.13s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  26% 0.25833459617740734/1 [00:03<00:11, 15.26s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  27% 0.26548462161849784/1 [00:03<00:11, 15.17s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  27% 0.2724380827862938/1 [00:03<00:11, 15.95s/it] [A
cudatoolkit-11.8.0   | 635.9 MB  | :  28% 0.28189773843859567/1 [00:03<00:10, 14.09s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  29% 0.2917505226374867/1 [00:03<00:09, 12.75s/it] [A
cudatoolkit-11.8.0   | 635.9 MB  | :  30% 0.3014558836314068/1 [00:03<00:08, 11.96s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  31% 0.3109892508861941/1 [00:03<00:07, 11.51s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  32% 0.31988378425277403/1 [00:03<00:10, 15.30s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  33% 0.3273286561038064/1 [00:04<00:10, 15.15s/it] [A
cudatoolkit-11.8.0   | 635.9 MB  | :  34% 0.3377711331225812/1 [00:04<00:08, 13.24s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  35% 0.3459776915326301/1 [00:04<00:08, 13.46s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  35% 0.3538648329985753/1 [00:04<00:08, 13.86s/it][A

cupy-12.0.0          | 34.8 MB   | : 100% 1.0/1 [00:04<00:00,  1.25it/s]               [A[A
cudatoolkit-11.8.0   | 635.9 MB  | :  37% 0.37044994355780586/1 [00:04<00:06, 10.32s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  38% 0.38236665262629/1 [00:04<00:05,  9.71s/it]   [A
cudatoolkit-11.8.0   | 635.9 MB  | :  40% 0.39607701068858725/1 [00:04<00:05,  8.89s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  41% 0.4134975194093198/1 [00:04<00:04,  7.68s/it] [A
cudatoolkit-11.8.0   | 635.9 MB  | :  43% 0.4299106362294176/1 [00:04<00:04,  7.14s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  44% 0.4441861165774368/1 [00:04<00:03,  7.19s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  46% 0.46059923339753456/1 [00:05<00:03,  6.83s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  48% 0.475390694962952/1 [00:05<00:03,  6.91s/it]  [A
cudatoolkit-11.8.0   | 635.9 MB  | :  49% 0.4899855922550749/1 [00:05<00:03,  7.66s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  51% 0.5050473296962724/1 [00:05<00:03,  7.35s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  52% 0.5228855374977559/1 [00:05<00:03,  6.76s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  54% 0.5379718454731153/1 [00:05<00:03,  7.38s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  55% 0.5538689810758147/1 [00:05<00:03,  7.05s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  57% 0.5701101041567798/1 [00:05<00:02,  6.78s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  59% 0.5851472710638155/1 [00:05<00:03,  7.38s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  60% 0.5990541933994074/1 [00:06<00:02,  7.48s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  62% 0.6155901628903142/1 [00:06<00:02,  7.02s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  63% 0.6307747530023209/1 [00:06<00:02,  6.89s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  65% 0.645467932431091/1 [00:06<00:02,  7.67s/it] [A
cudatoolkit-11.8.0   | 635.9 MB  | :  66% 0.6631341464934418/1 [00:06<00:02,  7.01s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  68% 0.6797192570526723/1 [00:06<00:02,  6.71s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  70% 0.6959603801336374/1 [00:06<00:01,  6.54s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  71% 0.7122506442829261/1 [00:06<00:01,  6.42s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  73% 0.7289831780471275/1 [00:06<00:01,  6.29s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  75% 0.7450523073889598/1 [00:07<00:01,  7.02s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  76% 0.7597209162835682/1 [00:07<00:01,  7.68s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  77% 0.7731855690042472/1 [00:07<00:01,  8.11s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  79% 0.7858393940975861/1 [00:07<00:01,  8.57s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  80% 0.7977561031660702/1 [00:07<00:01,  9.33s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  81% 0.8087145614022433/1 [00:07<00:01,  9.39s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  82% 0.8242677095266773/1 [00:07<00:01,  8.38s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  84% 0.8401402745952149/1 [00:07<00:01,  7.69s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  86% 0.8568973788935782/1 [00:08<00:01,  7.12s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  87% 0.8744407402851199/1 [00:08<00:00,  6.65s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  89% 0.8897481830679356/1 [00:08<00:00,  6.71s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  91% 0.9056453186706351/1 [00:08<00:00,  6.59s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  92% 0.9233115327329858/1 [00:08<00:00,  6.29s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  94% 0.9393315210064945/1 [00:08<00:00,  6.76s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  95% 0.9543686879135302/1 [00:08<00:00,  7.69s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  97% 0.9678824817025329/1 [00:08<00:00,  8.02s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | :  98% 0.9827967959387595/1 [00:08<00:00,  7.63s/it][A
cudatoolkit-11.8.0   | 635.9 MB  | : 100% 0.9987922136781062/1 [00:09<00:00,  7.20s/it][A
                                                                        
                                                                        [A

                                                                        [A[A
Preparing transaction: - done
Verifying transaction: | done
Executing transaction: - \ | / - \ By downloading and using the CUDA Toolkit conda packages, you accept the terms and conditions of the CUDA End User License Agreement (EULA): https://docs.nvidia.com/cuda/eula/index.html

done
Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
Collecting en-core-web-sm==3.5.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0-py3-none-any.whl (12.8 MB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m12.8/12.8 MB[0m [31m61.1 MB/s[0m eta [36m0:00:00[0m
[?25hRequirement already satisfied: spacy<3.6.0,>=3.5.0 in /usr/local/lib/python3.9/site-packages (from en-core-web-sm==3.5.0) (3.5.1)
Requirement already satisfied: smart-open<7.0.0,>=5.2.1 in /usr/local/lib/python3.9/site-packages (from spacy<3.6.0,>=3.5.0->en-core-web-sm==3.5.0) (5.2.1)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /usr/local/lib/python3.9/site-packages (from spacy<3.6.0,>=3.5.0->en-core-web-sm==3.5.0) (3.0.8)
Requirement already satisfied: pydantic!=1.8,!=1.8.1,<1.11.0,>=1.7.4 in /usr/local/lib/python3.9/site-packages (from spacy<3.6.0,>=3.5.0->en-core-web-sm==3.5.0) (1.10.7)
Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.9/site-packages (from spacy<3.6.0,>=3.5.0->en-core-web-sm==3.5.0) (23.0)
Requirement already satisfied: typer<0.8.0,>=0.3.0 in /usr/local/lib/python3.9/site-packages (from spacy<3.6.0,>=3.5.0->en-core-web-sm==3.5.0) (0.7.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /usr/local/lib/python3.9/site-packages (from spacy<3.6.0,>=3.5.0->en-core-web-sm==3.5.0) (2.28.2)
Requirement already satisfied: jinja2 in /usr/local/lib/python3.9/site-packages (from spacy<3.6.0,>=3.5.0->en-core-web-sm==3.5.0) (3.1.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /usr/local/lib/python3.9/site-packages (from spacy<3.6.0,>=3.5.0->en-core-web-sm==3.5.0) (4.65.0)
Requirement already satisfied: srsly<3.0.0,>=2.4.3 in /usr/local/lib/python3.9/site-packages (from spacy<3.6.0,>=3.5.0->en-core-web-sm==3.5.0) (2.4.6)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /usr/local/lib/python3.9/site-packages (from spacy<3.6.0,>=3.5.0->en-core-web-sm==3.5.0) (2.0.7)
Requirement already satisfied: langcodes<4.0.0,>=3.2.0 in /usr/local/lib/python3.9/site-packages (from spacy<3.6.0,>=3.5.0->en-core-web-sm==3.5.0) (3.3.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /usr/local/lib/python3.9/site-packages (from spacy<3.6.0,>=3.5.0->en-core-web-sm==3.5.0) (1.0.9)
Requirement already satisfied: pathy>=0.10.0 in /usr/local/lib/python3.9/site-packages (from spacy<3.6.0,>=3.5.0->en-core-web-sm==3.5.0) (0.10.1)
Requirement already satisfied: numpy>=1.15.0 in /usr/local/lib/python3.9/site-packages (from spacy<3.6.0,>=3.5.0->en-core-web-sm==3.5.0) (1.24.2)
Requirement already satisfied: spacy-legacy<3.1.0,>=3.0.11 in /usr/local/lib/python3.9/site-packages (from spacy<3.6.0,>=3.5.0->en-core-web-sm==3.5.0) (3.0.12)
Requirement already satisfied: wasabi<1.2.0,>=0.9.1 in /usr/local/lib/python3.9/site-packages (from spacy<3.6.0,>=3.5.0->en-core-web-sm==3.5.0) (1.1.1)
Requirement already satisfied: setuptools in /usr/local/lib/python3.9/site-packages (from spacy<3.6.0,>=3.5.0->en-core-web-sm==3.5.0) (65.6.3)
Requirement already satisfied: thinc<8.2.0,>=8.1.8 in /usr/local/lib/python3.9/site-packages (from spacy<3.6.0,>=3.5.0->en-core-web-sm==3.5.0) (8.1.9)
Requirement already satisfied: spacy-loggers<2.0.0,>=1.0.0 in /usr/local/lib/python3.9/site-packages (from spacy<3.6.0,>=3.5.0->en-core-web-sm==3.5.0) (1.0.4)
Requirement already satisfied: catalogue<2.1.0,>=2.0.6 in /usr/local/lib/python3.9/site-packages (from spacy<3.6.0,>=3.5.0->en-core-web-sm==3.5.0) (2.0.8)
Requirement already satisfied: typing-extensions>=4.2.0 in /usr/local/lib/python3.9/site-packages (from pydantic!=1.8,!=1.8.1,<1.11.0,>=1.7.4->spacy<3.6.0,>=3.5.0->en-core-web-sm==3.5.0) (4.5.0)
Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.9/site-packages (from requests<3.0.0,>=2.13.0->spacy<3.6.0,>=3.5.0->en-core-web-sm==3.5.0) (2.1.1)
Requirement already satisfied: urllib3<1.27,>=1.21.1 in /usr/local/lib/python3.9/site-packages (from requests<3.0.0,>=2.13.0->spacy<3.6.0,>=3.5.0->en-core-web-sm==3.5.0) (1.26.15)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.9/site-packages (from requests<3.0.0,>=2.13.0->spacy<3.6.0,>=3.5.0->en-core-web-sm==3.5.0) (2022.12.7)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.9/site-packages (from requests<3.0.0,>=2.13.0->spacy<3.6.0,>=3.5.0->en-core-web-sm==3.5.0) (3.4)
Requirement already satisfied: confection<1.0.0,>=0.0.1 in /usr/local/lib/python3.9/site-packages (from thinc<8.2.0,>=8.1.8->spacy<3.6.0,>=3.5.0->en-core-web-sm==3.5.0) (0.0.4)
Requirement already satisfied: blis<0.8.0,>=0.7.8 in /usr/local/lib/python3.9/site-packages (from thinc<8.2.0,>=8.1.8->spacy<3.6.0,>=3.5.0->en-core-web-sm==3.5.0) (0.7.9)
Requirement already satisfied: click<9.0.0,>=7.1.1 in /usr/local/lib/python3.9/site-packages (from typer<0.8.0,>=0.3.0->spacy<3.6.0,>=3.5.0->en-core-web-sm==3.5.0) (8.1.3)
Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.9/site-packages (from jinja2->spacy<3.6.0,>=3.5.0->en-core-web-sm==3.5.0) (2.1.2)
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-3.5.0
[33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
[0m[38;5;2m✔ Download and installation successful[0m
You can now load the package via spacy.load('en_core_web_sm')

~~~

## Linguistic Features

Spacy에는 POS Tagging, Morphology, Lemmatization, Dependency Parse, Named Entities, Entity Linking, Tokenization, Merging & splitting, Sentence segmentation 등의 다양한 특징을 가지고 있다.

시간적인 여유가 있다면 다음 리소스를 차근히 해보는 것도 좋다.
https://spacy.io/usage/linguistic-features#pos-tagging

우리는 spacytextblob를 이용한 감성분석(혹은 극성분석)을 진행하고자 한다. spacytextblob는 감성분석을 위햏 spacy 버전용 TextBlob라고 생각하면 ㄴ된다.

install spacytextblob

~~~python
!pip install spacytextblob
!python -m textblob.download_corpora

~~~

~~~python
!python -m textblob.download_corpora

~~~
Output:
~~~
[nltk_data] Downloading package brown to /root/nltk_data...
[nltk_data]   Package brown is already up-to-date!
[nltk_data] Downloading package punkt to /root/nltk_data...
[nltk_data]   Package punkt is already up-to-date!
[nltk_data] Downloading package wordnet to /root/nltk_data...
[nltk_data]   Package wordnet is already up-to-date!
[nltk_data] Downloading package averaged_perceptron_tagger to
[nltk_data]     /root/nltk_data...
[nltk_data]   Package averaged_perceptron_tagger is already up-to-
[nltk_data]       date!
[nltk_data] Downloading package conll2000 to /root/nltk_data...
[nltk_data]   Package conll2000 is already up-to-date!
[nltk_data] Downloading package movie_reviews to /root/nltk_data...
[nltk_data]   Package movie_reviews is already up-to-date!
Finished.

~~~

## add spacytextblob to the pipeline

~~~python
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob') ## add spacytextblob to pipeline
text = 'I had a really horrible day. It was the worst day ever! But every now and then I have a really good day that makes me happy.'

~~~

~~~python
doc = nlp(text)

~~~

## Sentiment analysis

~~~python
doc._.blob.polarity # 극성 [-1.0, 1.0]  [매우 부정, 매우 긍정]

~~~
Output:
~~~
-0.125

~~~

~~~python
doc._.blob.subjectivity # 주관성 [0.0, 1.0] [객관적, 매우 주관]

~~~
Output:
~~~
0.9

~~~

~~~python
doc._.blob.sentiment_assessments.assessments # 판단에 대한 평가 점수 ( 극성, 주관성, )

~~~
Output:
~~~
[(['really', 'horrible'], -1.0, 1.0, None),
 (['worst', '!'], -1.0, 1.0, None),
 (['really', 'good'], 0.7, 0.6000000000000001, None),
 (['happy'], 0.8, 1.0, None)]

~~~

#### Exercise
특정 유튜브 영상에서, 채팅 메세지의 긍정/부정에 대해서 통계화된 자료(평균 값, 비율 등)과 함께 적절한 예와 함께 설명하시오.

Consideration: 
채팅메세지는 문법에 맞지 않는 다양한 표현이 존재할 수 있다. 예를 들어, 약어 형태나 축약 단어형태가 존재할 수 있다. 이러한 부분을 해결하기 위해서는 어떻게 해야할지 고민해보자.

# 유튜브 자막 데이터 분석하기

유튜브 영상의 일부는 자막을 제공한다. 자막은 해당 컨텐츠를 이해하기 위한 중요한 부분정의 하나이다. 
 - 영화의

~~~python
YOUTUBE_ID_FOR_ANALYSIS = "2bP_KuBrXSc" #@param {type:"string"}

~~~

youtube_transcript_api는 api key 필요없이 직접적으로 자막을 가져올 수 있다.
주요기능은 다음과 같다.
- ID에 의한 자막 추출
- 언어별로 찾기
- 자동생성 된것인지 수동적으로 생성된것인지를 판별

install YouTubeTranscriptApi

~~~python
!pip install pip install youtube-transcript-api

~~~
Output:
~~~
Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
Requirement already satisfied: pip in /usr/local/lib/python3.9/site-packages (23.0.1)
Collecting install
  Downloading install-1.3.5-py3-none-any.whl (3.2 kB)
Collecting youtube-transcript-api
  Downloading youtube_transcript_api-0.5.0-py3-none-any.whl (23 kB)
Requirement already satisfied: requests in /usr/local/lib/python3.9/site-packages (from youtube-transcript-api) (2.28.2)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.9/site-packages (from requests->youtube-transcript-api) (3.4)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.9/site-packages (from requests->youtube-transcript-api) (2022.12.7)
Requirement already satisfied: urllib3<1.27,>=1.21.1 in /usr/local/lib/python3.9/site-packages (from requests->youtube-transcript-api) (1.26.15)
Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.9/site-packages (from requests->youtube-transcript-api) (2.1.1)
Installing collected packages: install, youtube-transcript-api
Successfully installed install-1.3.5 youtube-transcript-api-0.5.0
[33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
[0m

~~~

## message format of youtube-transcript-api

따라서 다음과 같이 메세지 정보를 파악할 수 있다. 참고: https://pypi.org/project/youtube-transcript-api/

```
print(
    transcript.video_id,
    transcript.language,
    transcript.language_code,
    # whether it has been manually created or generated by YouTube
    transcript.is_generated,
    # whether this transcript can be translated or not
    transcript.is_translatable,
    # a list of languages the transcript can be translated to
    transcript.translation_languages,
)
```

## call transcript api

~~~python
from youtube_transcript_api import YouTubeTranscriptApi

transcript = YouTubeTranscriptApi.get_transcript(YOUTUBE_ID_FOR_ANALYSIS)

~~~

해당 자막 시작하는 시간(start), 해당 자막의 내용(text), 해당 자막이 완료되는 종료시점(duration)으로 추출된다.
```
[
    {
        'text': 'Hey there',
        'start': 7.58,
        'duration': 6.13
    },
    {
        'text': 'how are you',
        'start': 14.08,
        'duration': 7.58
    },
    # ...
]
```

## print results

~~~python
for trans in transcript:
  print(trans)

~~~
Output:
~~~
{'text': 'hello and welcome back to another video', 'start': 0.0, 'duration': 3.54}
{'text': 'on this channel', 'start': 2.1, 'duration': 3.84}
{'text': "recently I've been studying active", 'start': 3.54, 'duration': 4.979}
{'text': 'variant neural networks and find them to', 'start': 5.94, 'duration': 5.279}
{'text': 'be a very interesting class of models', 'start': 8.519, 'duration': 4.74}
{'text': 'they integrate specific assumptions', 'start': 11.219, 'duration': 4.16}
{'text': 'about the data and can lead to a better', 'start': 13.259, 'duration': 4.381}
{'text': 'generalizability and are usually more', 'start': 15.379, 'duration': 5.441}
{'text': 'data efficient the theory can get quite', 'start': 17.64, 'duration': 5.46}
{'text': 'mathematical but in this video I will', 'start': 20.82, 'duration': 4.799}
{'text': 'try to keep it on a high level as this', 'start': 23.1, 'duration': 4.439}
{'text': 'is intended as an introduction to the', 'start': 25.619, 'duration': 4.441}
{'text': 'topic if you have questions or want me', 'start': 27.539, 'duration': 5.04}
{'text': 'to go deeper into specific things please', 'start': 30.06, 'duration': 4.92}
{'text': 'let me know in the comments I hope you', 'start': 32.579, 'duration': 4.381}
{'text': "enjoyed the following let's get started", 'start': 34.98, 'duration': 4.86}
{'text': "and before I forget I'm not an expert in", 'start': 36.96, 'duration': 4.98}
{'text': 'this field I just read a lot about it in', 'start': 39.84, 'duration': 4.14}
{'text': 'the last weeks and this video is', 'start': 41.94, 'duration': 4.56}
{'text': 'basically the summary of it', 'start': 43.98, 'duration': 4.38}
{'text': 'I want to start with some intuition', 'start': 46.5, 'duration': 4.079}
{'text': 'about equivariants in case you have', 'start': 48.36, 'duration': 4.859}
{'text': 'never come across this term a desirable', 'start': 50.579, 'duration': 5.46}
{'text': 'property for a neural network is that no', 'start': 53.219, 'duration': 4.201}
{'text': 'matter where some pattern of Interest', 'start': 56.039, 'duration': 4.02}
{'text': 'occurs in our data the model is able to', 'start': 57.42, 'duration': 3.9}
{'text': 'detect it', 'start': 60.059, 'duration': 3.48}
{'text': 'for example we pass this doc image', 'start': 61.32, 'duration': 3.78}
{'text': 'through the model and get high', 'start': 63.539, 'duration': 3.421}
{'text': 'activations for the area where the dog', 'start': 65.1, 'duration': 4.62}
{'text': 'is now the model should also respond', 'start': 66.96, 'duration': 5.04}
{'text': 'similarly if the dog occurs somewhere', 'start': 69.72, 'duration': 5.1}
{'text': 'else in the image in other words the', 'start': 72.0, 'duration': 4.92}
{'text': 'model should be insensitive to certain', 'start': 74.82, 'duration': 4.64}
{'text': 'transformations of the objects', 'start': 76.92, 'duration': 4.98}
{'text': 'equivariance simply means that when the', 'start': 79.46, 'duration': 5.199}
{'text': 'pattern in the input changes the output', 'start': 81.9, 'duration': 5.82}
{'text': 'changes in an equivalent proportion', 'start': 84.659, 'duration': 5.401}
{'text': 'for example the feature activation maps', 'start': 87.72, 'duration': 5.1}
{'text': 'in a CNN would be translated in a', 'start': 90.06, 'duration': 5.28}
{'text': 'similar fashion like this', 'start': 92.82, 'duration': 5.52}
{'text': 'okay so that is equivariance but you', 'start': 95.34, 'duration': 5.639}
{'text': 'might have also heard of invariance', 'start': 98.34, 'duration': 4.919}
{'text': 'I also want to get this out of the way', 'start': 100.979, 'duration': 5.28}
{'text': 'early in the case of invariance the', 'start': 103.259, 'duration': 6.061}
{'text': 'output does not change at all so it has', 'start': 106.259, 'duration': 5.161}
{'text': 'no variance to transformations of the', 'start': 109.32, 'duration': 5.1}
{'text': 'input the system basically produces the', 'start': 111.42, 'duration': 6.0}
{'text': 'exactly same response regardless of how', 'start': 114.42, 'duration': 5.4}
{'text': 'the input is transformed', 'start': 117.42, 'duration': 4.979}
{'text': 'in this example the fact that there is a', 'start': 119.82, 'duration': 4.86}
{'text': 'dock does not depend on where it is', 'start': 122.399, 'duration': 4.441}
{'text': 'located so the model prediction stays', 'start': 124.68, 'duration': 4.38}
{'text': 'the same in many neural networks', 'start': 126.84, 'duration': 4.68}
{'text': 'invariance comes from pooling layers', 'start': 129.06, 'duration': 4.56}
{'text': 'that aggregate some activations at the', 'start': 131.52, 'duration': 4.38}
{'text': 'end of the network for example when you', 'start': 133.62, 'duration': 4.199}
{'text': "apply Max pooling it doesn't matter", 'start': 135.9, 'duration': 4.44}
{'text': 'which neuron spits out the max value but', 'start': 137.819, 'duration': 5.581}
{'text': 'just that it exists these so-called set', 'start': 140.34, 'duration': 6.18}
{'text': 'functions like Max means sum and so on', 'start': 143.4, 'duration': 6.3}
{'text': 'are a simple way to achieve invariance', 'start': 146.52, 'duration': 5.4}
{'text': 'one important point is that Equity', 'start': 149.7, 'duration': 4.619}
{'text': 'variances and invariances are always', 'start': 151.92, 'duration': 4.2}
{'text': 'defined with respect to some', 'start': 154.319, 'duration': 3.361}
{'text': 'transformation class', 'start': 156.12, 'duration': 4.5}
{'text': 'in this case it is translation onto the', 'start': 157.68, 'duration': 5.279}
{'text': 'images because we have been moving the', 'start': 160.62, 'duration': 5.399}
{'text': 'dog in the two-dimensional pixel space', 'start': 162.959, 'duration': 4.801}
{'text': 'that means no matter how we translate', 'start': 166.019, 'duration': 4.44}
{'text': 'the object of Interest the model output', 'start': 167.76, 'duration': 4.68}
{'text': 'should either change equally for', 'start': 170.459, 'duration': 4.741}
{'text': 'equivariance or stay constant for', 'start': 172.44, 'duration': 4.019}
{'text': 'invariance', 'start': 175.2, 'duration': 3.66}
{'text': 'regular fully connected neural networks', 'start': 176.459, 'duration': 4.5}
{'text': "don't have this property as they are", 'start': 178.86, 'duration': 4.86}
{'text': 'sensitive to the order of the inputs it', 'start': 180.959, 'duration': 4.681}
{'text': 'turns out however that there is already', 'start': 183.72, 'duration': 4.44}
{'text': 'a model architecture that is translation', 'start': 185.64, 'duration': 4.8}
{'text': 'equivariant which are convolutional', 'start': 188.16, 'duration': 4.799}
{'text': 'neural networks in fact one of the key', 'start': 190.44, 'duration': 5.34}
{'text': "motivations of CNN's besides efficiency", 'start': 192.959, 'duration': 5.341}
{'text': 'was equivariance with respect to', 'start': 195.78, 'duration': 4.86}
{'text': 'translations which is achieved by', 'start': 198.3, 'duration': 3.78}
{'text': 'convolutions', 'start': 200.64, 'duration': 3.54}
{'text': 'effectively convolutions are nothing', 'start': 202.08, 'duration': 5.28}
{'text': 'else but translations of filters applied', 'start': 204.18, 'duration': 6.24}
{'text': 'on the image or in other words the value', 'start': 207.36, 'duration': 6.36}
{'text': 'of the feature map is computed as inner', 'start': 210.42, 'duration': 5.76}
{'text': 'product between the inputs and the', 'start': 213.72, 'duration': 4.86}
{'text': 'filter shifted by some value', 'start': 216.18, 'duration': 4.86}
{'text': 'sometimes equivariants and invariants', 'start': 218.58, 'duration': 4.98}
{'text': 'are used exchangeably especially in the', 'start': 221.04, 'duration': 5.4}
{'text': 'context of conf Nets but by now you', 'start': 223.56, 'duration': 5.039}
{'text': 'should know that equivariance is related', 'start': 226.44, 'duration': 4.26}
{'text': 'to the changes in the outputs of the', 'start': 228.599, 'duration': 4.5}
{'text': 'neurons and invariance comes from', 'start': 230.7, 'duration': 4.319}
{'text': 'pooling operations at the end of the', 'start': 233.099, 'duration': 4.081}
{'text': 'network it is also possible to Define', 'start': 235.019, 'duration': 4.321}
{'text': 'this more formally with very simple', 'start': 237.18, 'duration': 3.839}
{'text': 'expressions', 'start': 239.34, 'duration': 4.74}
{'text': 'a function in our case the model is said', 'start': 241.019, 'duration': 5.341}
{'text': 'to be equivalent when applying some', 'start': 244.08, 'duration': 4.2}
{'text': 'transformation T for example', 'start': 246.36, 'duration': 5.34}
{'text': 'translations on an image on input X', 'start': 248.28, 'duration': 5.28}
{'text': 'which is the image', 'start': 251.7, 'duration': 4.68}
{'text': 'if the output of our model changes', 'start': 253.56, 'duration': 4.919}
{'text': 'equivalently through some other', 'start': 256.38, 'duration': 5.22}
{'text': 'transformation T Prime for invariance on', 'start': 258.479, 'duration': 4.861}
{'text': 'the other hand the output stays', 'start': 261.6, 'duration': 3.72}
{'text': 'unaffected', 'start': 263.34, 'duration': 3.96}
{'text': 'a common way to visualize this', 'start': 265.32, 'duration': 4.68}
{'text': 'graphically is with this diagram', 'start': 267.3, 'duration': 4.86}
{'text': 'this is the case for equivariance', 'start': 270.0, 'duration': 4.32}
{'text': 'applying some transformation and then', 'start': 272.16, 'duration': 3.479}
{'text': 'passing through the neural network', 'start': 274.32, 'duration': 3.9}
{'text': 'should lead to the same result as first', 'start': 275.639, 'duration': 4.381}
{'text': 'going through the model and then', 'start': 278.22, 'duration': 4.02}
{'text': 'applying a transformation', 'start': 280.02, 'duration': 5.16}
{'text': 'or if you like to think of spaces I also', 'start': 282.24, 'duration': 5.94}
{'text': 'find this chart nice invariance maps to', 'start': 285.18, 'duration': 4.64}
{'text': 'the same point in the Target space', 'start': 288.18, 'duration': 3.78}
{'text': 'independent of the transformed starting', 'start': 289.82, 'duration': 5.439}
{'text': 'points and equivariance transforms in a', 'start': 291.96, 'duration': 5.16}
{'text': 'predictable way in the Target space', 'start': 295.259, 'duration': 4.5}
{'text': 'after transformations in the origin', 'start': 297.12, 'duration': 5.28}
{'text': 'space sometimes invariance is also', 'start': 299.759, 'duration': 4.801}
{'text': 'referred to as dropping spatial', 'start': 302.4, 'duration': 4.32}
{'text': 'information which can be seen nicely', 'start': 304.56, 'duration': 4.62}
{'text': 'here to wrap this up we can design', 'start': 306.72, 'duration': 5.4}
{'text': 'special architectures like cnns that', 'start': 309.18, 'duration': 4.98}
{'text': 'allow us to process input data more', 'start': 312.12, 'duration': 4.32}
{'text': 'efficiently by exploiting symmetries in', 'start': 314.16, 'duration': 5.34}
{'text': 'the data symmetries here means that some', 'start': 316.44, 'duration': 5.52}
{'text': 'properties of interest in our data which', 'start': 319.5, 'duration': 4.979}
{'text': 'were transformed somehow stay unchanged', 'start': 321.96, 'duration': 4.079}
{'text': 'after the transformation', 'start': 324.479, 'duration': 4.081}
{'text': 'these invariant models take advantage of', 'start': 326.039, 'duration': 4.801}
{'text': 'the fact that the properties in the data', 'start': 328.56, 'duration': 4.68}
{'text': 'are invariant to the Symmetry', 'start': 330.84, 'duration': 5.22}
{'text': 'Transformations but the data itself is', 'start': 333.24, 'duration': 3.72}
{'text': 'not', 'start': 336.06, 'duration': 3.54}
{'text': 'most people think of symmetry as simply', 'start': 336.96, 'duration': 5.7}
{'text': "mirroring along an axis that's the most", 'start': 339.6, 'duration': 5.159}
{'text': 'common type of symmetry so-called', 'start': 342.66, 'duration': 4.08}
{'text': 'reflection symmetry', 'start': 344.759, 'duration': 4.38}
{'text': 'but this mathematical concept can also', 'start': 346.74, 'duration': 5.519}
{'text': 'be defined Beyond Simple Reflections in', 'start': 349.139, 'duration': 4.861}
{'text': 'fact there exist many other symmetries', 'start': 352.259, 'duration': 3.801}
{'text': 'such as translational symmetry', 'start': 354.0, 'duration': 4.86}
{'text': 'rotational symmetry or permutation', 'start': 356.06, 'duration': 4.9}
{'text': 'symmetry', 'start': 358.86, 'duration': 5.7}
{'text': 'generally a symmetry transformation is a', 'start': 360.96, 'duration': 4.98}
{'text': 'transformation that leaves the', 'start': 364.56, 'duration': 4.44}
{'text': 'properties of an object unchanged in our', 'start': 365.94, 'duration': 4.92}
{'text': 'previous example for instance the fact', 'start': 369.0, 'duration': 4.08}
{'text': 'that there was a dog in the image', 'start': 370.86, 'duration': 4.5}
{'text': 'so we want the invariant model to ignore', 'start': 373.08, 'duration': 4.04}
{'text': 'the changes through the transformation', 'start': 375.36, 'duration': 5.04}
{'text': 'and just focus on the properties of the', 'start': 377.12, 'duration': 4.96}
{'text': 'object of Interest', 'start': 380.4, 'duration': 4.98}
{'text': 'going back to our initial example cnns', 'start': 382.08, 'duration': 5.52}
{'text': 'are known to be equivariant with respect', 'start': 385.38, 'duration': 5.759}
{'text': 'to translation but not to rotation so', 'start': 387.6, 'duration': 4.74}
{'text': 'why is this', 'start': 391.139, 'duration': 3.661}
{'text': 'the learn filters do not respond to', 'start': 392.34, 'duration': 5.1}
{'text': 'rotated versions of objects as long as', 'start': 394.8, 'duration': 4.56}
{'text': "they weren't part of the training data", 'start': 397.44, 'duration': 4.199}
{'text': "that's because the convolutional Fitters", 'start': 399.36, 'duration': 5.22}
{'text': 'are translated in the pixel space but', 'start': 401.639, 'duration': 5.701}
{'text': 'always kept at the same rotation degree', 'start': 404.58, 'duration': 4.98}
{'text': 'throughout this video series we will get', 'start': 407.34, 'duration': 3.84}
{'text': 'familiar with ways to achieve', 'start': 409.56, 'duration': 4.32}
{'text': 'equivariance with respect to also other', 'start': 411.18, 'duration': 5.579}
{'text': 'symmetry groups incorporating knowledge', 'start': 413.88, 'duration': 4.98}
{'text': 'about underlying symmetries in neural', 'start': 416.759, 'duration': 5.041}
{'text': 'networks is a so-called inductive bias', 'start': 418.86, 'duration': 5.339}
{'text': 'that means we expect the data to follow', 'start': 421.8, 'duration': 4.44}
{'text': 'the assumptions that the model was', 'start': 424.199, 'duration': 3.481}
{'text': 'designed for', 'start': 426.24, 'duration': 3.78}
{'text': 'this makes the model more efficient and', 'start': 427.68, 'duration': 4.739}
{'text': 'usually allows better performance and', 'start': 430.02, 'duration': 4.26}
{'text': 'reduces the amount of data that is', 'start': 432.419, 'duration': 3.06}
{'text': 'needed', 'start': 434.28, 'duration': 3.84}
{'text': 'we need to be careful however as we must', 'start': 435.479, 'duration': 5.581}
{'text': 'be sure that the data follows our rules', 'start': 438.12, 'duration': 4.74}
{'text': 'by introducing assumptions about', 'start': 441.06, 'duration': 4.38}
{'text': 'symmetries the flexibility of our', 'start': 442.86, 'duration': 5.399}
{'text': 'Network decreases as we can only operate', 'start': 445.44, 'duration': 6.12}
{'text': 'on data that aligns with our assumptions', 'start': 448.259, 'duration': 5.101}
{'text': "there's a pretty good example for this", 'start': 451.56, 'duration': 5.039}
{'text': 'in stereochemistry called chirality', 'start': 453.36, 'duration': 5.76}
{'text': 'the properties of molecules are usually', 'start': 456.599, 'duration': 5.04}
{'text': 'invariant to rotation and translation', 'start': 459.12, 'duration': 4.859}
{'text': 'that means no matter how we arrange them', 'start': 461.639, 'duration': 4.321}
{'text': 'in the space the properties stay', 'start': 463.979, 'duration': 3.421}
{'text': 'unchanged', 'start': 465.96, 'duration': 4.38}
{'text': 'they are however so-called chiral', 'start': 467.4, 'duration': 4.68}
{'text': 'molecules which can change their', 'start': 470.34, 'duration': 4.139}
{'text': 'properties on reflection', 'start': 472.08, 'duration': 5.459}
{'text': 'if we now use reflection symmetry as an', 'start': 474.479, 'duration': 5.761}
{'text': 'inductive bias we need to be aware that', 'start': 477.539, 'duration': 4.681}
{'text': 'we will not be able to separate these', 'start': 480.24, 'duration': 3.12}
{'text': 'molecules', 'start': 482.22, 'duration': 3.06}
{'text': "so it's always important to make sure", 'start': 483.36, 'duration': 3.899}
{'text': 'that the Symmetry assumptions of the', 'start': 485.28, 'duration': 4.199}
{'text': 'model align with what we find in the', 'start': 487.259, 'duration': 3.121}
{'text': 'data', 'start': 489.479, 'duration': 3.301}
{'text': "talking about data what's wrong with", 'start': 490.38, 'duration': 4.379}
{'text': 'data augmentations', 'start': 492.78, 'duration': 4.919}
{'text': 'data augmentation is the most common way', 'start': 494.759, 'duration': 5.401}
{'text': 'to make the model insensitive to', 'start': 497.699, 'duration': 4.68}
{'text': 'different Transformations and therefore', 'start': 500.16, 'duration': 4.379}
{'text': 'most of you have probably already had a', 'start': 502.379, 'duration': 4.681}
{'text': 'touch point with equivariance', 'start': 504.539, 'duration': 5.16}
{'text': 'by augmenting the data we can make the', 'start': 507.06, 'duration': 4.74}
{'text': 'models learn to be equivalent with', 'start': 509.699, 'duration': 4.501}
{'text': 'respect to some symmetry class for', 'start': 511.8, 'duration': 4.2}
{'text': 'example in this case we can simply train', 'start': 514.2, 'duration': 4.68}
{'text': 'a model not only on the dog image but', 'start': 516.0, 'duration': 5.7}
{'text': 'also on many rotated versions of it', 'start': 518.88, 'duration': 5.099}
{'text': 'this way the model most likely produces', 'start': 521.7, 'duration': 4.079}
{'text': 'very similar outputs for different', 'start': 523.979, 'duration': 4.98}
{'text': 'rotations of the same image but there', 'start': 525.779, 'duration': 5.461}
{'text': 'are also downsides of data augmentation', 'start': 528.959, 'duration': 5.401}
{'text': 'on one hand the performance of truly', 'start': 531.24, 'duration': 6.12}
{'text': 'equivariant models is usually better as', 'start': 534.36, 'duration': 5.159}
{'text': 'shown in different studies for example', 'start': 537.36, 'duration': 5.099}
{'text': 'quoting this paper our model results', 'start': 539.519, 'duration': 5.88}
{'text': 'indicate that equivariant models possess', 'start': 542.459, 'duration': 5.06}
{'text': 'in inherent advantage over', 'start': 545.399, 'duration': 4.741}
{'text': 'non-equivariant ones which cannot be', 'start': 547.519, 'duration': 5.801}
{'text': 'overcome by data augmentation', 'start': 550.14, 'duration': 6.12}
{'text': 'secondly data augmentation is not as', 'start': 553.32, 'duration': 5.699}
{'text': 'efficient as equivariant models and also', 'start': 556.26, 'duration': 5.04}
{'text': "doesn't work properly for all symmetry", 'start': 559.019, 'duration': 3.481}
{'text': 'groups', 'start': 561.3, 'duration': 3.42}
{'text': 'another common argument is that data', 'start': 562.5, 'duration': 4.26}
{'text': 'augmentation can only be applied on the', 'start': 564.72, 'duration': 4.799}
{'text': 'input layer whereas with these models we', 'start': 566.76, 'duration': 4.74}
{'text': 'can introduce equivariants in every', 'start': 569.519, 'duration': 3.901}
{'text': "layer let's wrap up some of the", 'start': 571.5, 'duration': 4.2}
{'text': 'motivations for Designing equivariant', 'start': 573.42, 'duration': 4.5}
{'text': 'neural networks first of all we need', 'start': 575.7, 'duration': 4.56}
{'text': 'much less data and can get rid of data', 'start': 577.92, 'duration': 3.72}
{'text': 'augmentations', 'start': 580.26, 'duration': 3.36}
{'text': 'then we are also able to introduce', 'start': 581.64, 'duration': 4.5}
{'text': 'equivariants in all layers not just the', 'start': 583.62, 'duration': 5.48}
{'text': 'input layer finally it leads to better', 'start': 586.14, 'duration': 5.759}
{'text': 'generalizability and less complex models', 'start': 589.1, 'duration': 5.38}
{'text': 'due to weight sharing just a quick check', 'start': 591.899, 'duration': 4.62}
{'text': 'at this point of the video you should be', 'start': 594.48, 'duration': 4.32}
{'text': 'familiar with what equivariants and', 'start': 596.519, 'duration': 4.5}
{'text': 'invariants are what symmetry', 'start': 598.8, 'duration': 4.92}
{'text': 'Transformations do on a high level and', 'start': 601.019, 'duration': 4.621}
{'text': 'finally what the motivation of', 'start': 603.72, 'duration': 4.799}
{'text': "equivariant neural networks is let's", 'start': 605.64, 'duration': 4.98}
{'text': 'move on with the final part of this', 'start': 608.519, 'duration': 4.681}
{'text': 'video first there is a quite interesting', 'start': 610.62, 'duration': 4.44}
{'text': 'article on naturally occurring', 'start': 613.2, 'duration': 4.199}
{'text': 'equivariants in neural networks that I', 'start': 615.06, 'duration': 4.8}
{'text': 'can highly recommend it talks about the', 'start': 617.399, 'duration': 3.901}
{'text': 'symmetries and the weights of the', 'start': 619.86, 'duration': 3.539}
{'text': 'networks basically there can be', 'start': 621.3, 'duration': 3.96}
{'text': 'different variants of a neuron that', 'start': 623.399, 'duration': 4.201}
{'text': 'respond to transformed versions of an', 'start': 625.26, 'duration': 4.98}
{'text': 'input but for more details please have a', 'start': 627.6, 'duration': 4.679}
{'text': 'look at the linked website in the video', 'start': 630.24, 'duration': 3.659}
{'text': 'description', 'start': 632.279, 'duration': 3.781}
{'text': 'one of the conclusions of this article', 'start': 633.899, 'duration': 5.161}
{'text': 'however is that natural equivariance has', 'start': 636.06, 'duration': 5.459}
{'text': 'its limits and it can be pretty useful', 'start': 639.06, 'duration': 4.62}
{'text': 'to develop equivariant architectures', 'start': 641.519, 'duration': 5.401}
{'text': 'just like it was done with cnns and this', 'start': 643.68, 'duration': 5.159}
{'text': 'brings us to our first paper in this', 'start': 646.92, 'duration': 4.26}
{'text': 'video essentially the idea of this paper', 'start': 648.839, 'duration': 4.861}
{'text': 'is to extend the notion of equivariance', 'start': 651.18, 'duration': 5.52}
{'text': 'in cnns to other symmetry classes Beyond', 'start': 653.7, 'duration': 5.46}
{'text': 'translation the researchers that', 'start': 656.7, 'duration': 4.74}
{'text': 'addressed this were Taco Cohen and', 'start': 659.16, 'duration': 5.76}
{'text': 'maxwelling in a paper from 2016 called', 'start': 661.44, 'duration': 6.06}
{'text': 'group equivariant convolutional neural', 'start': 664.92, 'duration': 3.78}
{'text': 'networks', 'start': 667.5, 'duration': 3.36}
{'text': 'you can say that this is the foundation', 'start': 668.7, 'duration': 5.1}
{'text': 'of a lot of follow-up research', 'start': 670.86, 'duration': 5.039}
{'text': 'they present a general mathematical', 'start': 673.8, 'duration': 5.099}
{'text': 'framework based on group theory on how', 'start': 675.899, 'duration': 5.221}
{'text': 'to extend classical convolutions to', 'start': 678.899, 'duration': 4.081}
{'text': 'different symmetry groups', 'start': 681.12, 'duration': 4.14}
{'text': 'these more General convolutions are then', 'start': 682.98, 'duration': 4.799}
{'text': 'called group convolutions', 'start': 685.26, 'duration': 4.5}
{'text': "in order to understand what's going on", 'start': 687.779, 'duration': 4.381}
{'text': 'we need to talk a bit about group Theory', 'start': 689.76, 'duration': 4.98}
{'text': 'and then in the next video we will', 'start': 692.16, 'duration': 4.679}
{'text': 'investigate group convolutions in more', 'start': 694.74, 'duration': 4.92}
{'text': 'depth a group is mathematically defined', 'start': 696.839, 'duration': 6.12}
{'text': 'as a set that is equipped with a binary', 'start': 699.66, 'duration': 6.299}
{'text': 'operator denoted with a point here the', 'start': 702.959, 'duration': 5.101}
{'text': 'set is just a collection of functions', 'start': 705.959, 'duration': 4.44}
{'text': 'for example different translations or', 'start': 708.06, 'duration': 4.8}
{'text': 'rotations and the group operation', 'start': 710.399, 'duration': 5.761}
{'text': 'operates on elements of this set such as', 'start': 712.86, 'duration': 6.24}
{'text': 'adding them or multiplying them so the', 'start': 716.16, 'duration': 5.04}
{'text': 'group operator tells us how to compose', 'start': 719.1, 'duration': 5.46}
{'text': 'elements of the set and the output of', 'start': 721.2, 'duration': 5.94}
{'text': 'this operation must be another member of', 'start': 724.56, 'duration': 4.86}
{'text': 'the group we are in particular', 'start': 727.14, 'duration': 4.8}
{'text': 'interested in symmetry groups therefore', 'start': 729.42, 'duration': 4.8}
{'text': 'each of the elements in this set is a', 'start': 731.94, 'duration': 4.8}
{'text': "symmetry transformation finally it's", 'start': 734.22, 'duration': 5.28}
{'text': 'called a group action if the group acts', 'start': 736.74, 'duration': 5.039}
{'text': 'on some space for example the space of', 'start': 739.5, 'duration': 6.0}
{'text': 'pixels so we rotate or transform pixels', 'start': 741.779, 'duration': 5.821}
{'text': 'with the elements in the group', 'start': 745.5, 'duration': 4.68}
{'text': 'you might wonder why we talk about these', 'start': 747.6, 'duration': 4.859}
{'text': 'abstract mathematical Concepts here', 'start': 750.18, 'duration': 4.56}
{'text': "that's because active variances are", 'start': 752.459, 'duration': 4.5}
{'text': 'typically defined with respect to one of', 'start': 754.74, 'duration': 3.42}
{'text': 'these groups', 'start': 756.959, 'duration': 2.94}
{'text': "let's have a look at an example and", 'start': 758.16, 'duration': 4.26}
{'text': "things will get much more clear let's", 'start': 759.899, 'duration': 5.041}
{'text': 'say our set of Transformations consists', 'start': 762.42, 'duration': 5.82}
{'text': 'of different rotations that means each', 'start': 764.94, 'duration': 5.459}
{'text': 'element in our set is one possible', 'start': 768.24, 'duration': 4.7}
{'text': 'rotation such as 45 degrees', 'start': 770.399, 'duration': 6.901}
{'text': '180 degrees 300 degrees and also no', 'start': 772.94, 'duration': 7.48}
{'text': 'rotation meaning zero degrees with the', 'start': 777.3, 'duration': 5.339}
{'text': 'group operator we can combine these', 'start': 780.42, 'duration': 4.859}
{'text': 'Transformations such as first rotating', 'start': 782.639, 'duration': 6.0}
{'text': 'by 45 degrees and then by 180 degrees', 'start': 785.279, 'duration': 6.421}
{'text': 'now typically when operating on Spaces', 'start': 788.639, 'duration': 6.061}
{'text': 'such as the 2D pixel space we can also', 'start': 791.7, 'duration': 4.8}
{'text': 'use a different more mathematical', 'start': 794.7, 'duration': 4.379}
{'text': 'notation for the transformation instead', 'start': 796.5, 'duration': 5.76}
{'text': 'of just saying Rotation by X degrees', 'start': 799.079, 'duration': 5.461}
{'text': 'in the case of rotation we can for', 'start': 802.26, 'duration': 5.4}
{'text': 'example utilize rotation matrices', 'start': 804.54, 'duration': 5.52}
{'text': '45 degrees then corresponds to', 'start': 807.66, 'duration': 4.619}
{'text': 'multiplying some image with this Matrix', 'start': 810.06, 'duration': 5.219}
{'text': 'and 180 degrees corresponds to this', 'start': 812.279, 'duration': 4.081}
{'text': 'Matrix', 'start': 815.279, 'duration': 3.721}
{'text': 'this notation is more precise and tells', 'start': 816.36, 'duration': 4.68}
{'text': 'us what actually happens under the hood', 'start': 819.0, 'duration': 4.62}
{'text': 'it also allows us to parameterize', 'start': 821.04, 'duration': 5.16}
{'text': 'different sets of rotations with just', 'start': 823.62, 'duration': 5.339}
{'text': 'one Matrix we will come back to this in', 'start': 826.2, 'duration': 5.04}
{'text': 'the second video a proper group of', 'start': 828.959, 'duration': 4.68}
{'text': 'Transformations needs to fulfill four', 'start': 831.24, 'duration': 5.039}
{'text': "properties which I've added here in very", 'start': 833.639, 'duration': 5.64}
{'text': 'simple language closure means that the', 'start': 836.279, 'duration': 4.981}
{'text': 'result of some Transformations never', 'start': 839.279, 'duration': 3.901}
{'text': "leaves the group so it's another group", 'start': 841.26, 'duration': 4.92}
{'text': 'member for example if you rotate by some', 'start': 843.18, 'duration': 5.04}
{'text': 'degree you will still arrive at some', 'start': 846.18, 'duration': 4.94}
{'text': 'rotation that is also part of the group', 'start': 848.22, 'duration': 5.52}
{'text': 'associativity means that the composition', 'start': 851.12, 'duration': 5.44}
{'text': 'of elements can be grouped furthermore', 'start': 853.74, 'duration': 5.52}
{'text': 'an identity element needs to exist and', 'start': 856.56, 'duration': 4.32}
{'text': 'there needs to be an inverse element', 'start': 859.26, 'duration': 4.259}
{'text': 'that brings you back to the identity I', 'start': 860.88, 'duration': 4.62}
{'text': "won't go further into detail regarding", 'start': 863.519, 'duration': 4.62}
{'text': "these definitions but we'll quickly talk", 'start': 865.5, 'duration': 5.04}
{'text': 'about a visual way how to check if some', 'start': 868.139, 'duration': 5.101}
{'text': "group is valid let's say we have these", 'start': 870.54, 'duration': 5.22}
{'text': 'four possible rotation degrees and the', 'start': 873.24, 'duration': 5.159}
{'text': 'matrix multiplication operation which is', 'start': 875.76, 'duration': 4.5}
{'text': "performed on the matrices we've just", 'start': 878.399, 'duration': 2.761}
{'text': 'seen', 'start': 880.26, 'duration': 4.259}
{'text': 'do these elements form a valid group', 'start': 881.16, 'duration': 5.76}
{'text': 'one way to check this is to build a', 'start': 884.519, 'duration': 5.041}
{'text': 'Kaylee table which works by arranging', 'start': 886.92, 'duration': 4.68}
{'text': "all possible products of the group's", 'start': 889.56, 'duration': 4.26}
{'text': 'elements in a squared Matrix', 'start': 891.6, 'duration': 4.739}
{'text': 'we can for example easily identify the', 'start': 893.82, 'duration': 5.22}
{'text': 'identity element which is rotation by', 'start': 896.339, 'duration': 5.401}
{'text': 'zero degrees as it will always keep the', 'start': 899.04, 'duration': 5.039}
{'text': 'elements unaffected as you can see in', 'start': 901.74, 'duration': 5.219}
{'text': 'the first row and column when continuing', 'start': 904.079, 'duration': 5.82}
{'text': 'we can see that the attribute closure is', 'start': 906.959, 'duration': 5.101}
{'text': 'not fulfilled for this set of elements', 'start': 909.899, 'duration': 5.88}
{'text': 'because if we apply 45 degrees rotation', 'start': 912.06, 'duration': 6.779}
{'text': 'twice we end up with 90 degrees which is', 'start': 915.779, 'duration': 5.881}
{'text': "not part of our group therefore it's not", 'start': 918.839, 'duration': 4.86}
{'text': 'a valid group definition', 'start': 921.66, 'duration': 4.2}
{'text': 'this was just a very high level overview', 'start': 923.699, 'duration': 4.32}
{'text': 'in case you want to dig deeper into this', 'start': 925.86, 'duration': 3.96}
{'text': 'topic I can highly recommend the three', 'start': 928.019, 'duration': 4.861}
{'text': 'blue one Brown video on group theory in', 'start': 929.82, 'duration': 5.34}
{'text': 'the world of group Theory popular groups', 'start': 932.88, 'duration': 4.68}
{'text': "have special names let's have a look at", 'start': 935.16, 'duration': 5.46}
{'text': 'a few examples translation in 2D is', 'start': 937.56, 'duration': 6.0}
{'text': 'typically denoted with t this group is', 'start': 940.62, 'duration': 5.159}
{'text': 'defined over all possible translations', 'start': 943.56, 'duration': 5.579}
{'text': 'in r squared for example we shift some', 'start': 945.779, 'duration': 5.401}
{'text': 'objects to a different position on an', 'start': 949.139, 'duration': 4.981}
{'text': 'image we can add 90 degree rotations to', 'start': 951.18, 'duration': 5.94}
{'text': "this group and we'll end up in P4", 'start': 954.12, 'duration': 5.159}
{'text': "so it's a combination of 90 degree", 'start': 957.12, 'duration': 4.86}
{'text': 'rotations and some translation like', 'start': 959.279, 'duration': 4.86}
{'text': 'shown in this example', 'start': 961.98, 'duration': 4.38}
{'text': 'when operating in three dimensions the', 'start': 964.139, 'duration': 5.341}
{'text': 'group of rotations is called SO3 which', 'start': 966.36, 'duration': 5.279}
{'text': 'stands for special orthogonal group', 'start': 969.48, 'duration': 5.52}
{'text': 'rotations in 3D can be represented using', 'start': 971.639, 'duration': 6.0}
{'text': 'three by three matrices we can also add', 'start': 975.0, 'duration': 6.12}
{'text': 'translation here and end up in se3 the', 'start': 977.639, 'duration': 5.64}
{'text': 'special euclidean group', 'start': 981.12, 'duration': 5.1}
{'text': 'in most of these symmetry groups in 2D', 'start': 983.279, 'duration': 6.0}
{'text': 'or 3D the binary operator is matrix', 'start': 986.22, 'duration': 4.44}
{'text': 'multiplication', 'start': 989.279, 'duration': 3.781}
{'text': 'there are many many other groups which', 'start': 990.66, 'duration': 4.919}
{'text': 'are systematically categorized mainly by', 'start': 993.06, 'duration': 3.74}
{'text': 'how you can represent them', 'start': 995.579, 'duration': 3.781}
{'text': 'mathematically building architectures', 'start': 996.8, 'duration': 4.42}
{'text': 'that consider these different symmetry', 'start': 999.36, 'duration': 4.32}
{'text': 'groups have shown promising results and', 'start': 1001.22, 'duration': 4.44}
{'text': 'here I wanted to summarize some of the', 'start': 1003.68, 'duration': 4.5}
{'text': 'most impressive applications', 'start': 1005.66, 'duration': 4.859}
{'text': 'group convolutions have shown to produce', 'start': 1008.18, 'duration': 4.68}
{'text': 'better results on many Medical Imaging', 'start': 1010.519, 'duration': 4.641}
{'text': 'data sets because they for example', 'start': 1012.86, 'duration': 4.74}
{'text': 'incorporate rotational and translational', 'start': 1015.16, 'duration': 4.359}
{'text': 'equivariants', 'start': 1017.6, 'duration': 4.979}
{'text': "there's a paper about gcnn supplied on", 'start': 1019.519, 'duration': 6.121}
{'text': 'lung CT scans for detecting cancer and', 'start': 1022.579, 'duration': 5.34}
{'text': 'the authors show that gcnns not only', 'start': 1025.64, 'duration': 4.799}
{'text': 'perform better but are also much more', 'start': 1027.919, 'duration': 5.28}
{'text': 'data efficient one key contribution of', 'start': 1030.439, 'duration': 4.86}
{'text': 'the Improvement of alpha Fall 2 over', 'start': 1033.199, 'duration': 5.521}
{'text': 'Alpha folds is a se3 equivariant', 'start': 1035.299, 'duration': 5.581}
{'text': 'Transformer that became part of the', 'start': 1038.72, 'duration': 5.099}
{'text': 'architecture equivariants also plays an', 'start': 1040.88, 'duration': 4.919}
{'text': 'important role in a lot of the protein', 'start': 1043.819, 'duration': 4.14}
{'text': 'ligand binding models that were recently', 'start': 1045.799, 'duration': 4.681}
{'text': 'published basically this topic is about', 'start': 1047.959, 'duration': 4.501}
{'text': 'if some drug medicine fits into', 'start': 1050.48, 'duration': 4.559}
{'text': 'so-called pockets of a protein and', 'start': 1052.46, 'duration': 4.62}
{'text': 'apparently novel methods seem to do', 'start': 1055.039, 'duration': 4.441}
{'text': 'quite well generally many molecular', 'start': 1057.08, 'duration': 4.44}
{'text': 'applications seem to benefit from', 'start': 1059.48, 'duration': 4.26}
{'text': 'equivariant Prius', 'start': 1061.52, 'duration': 4.44}
{'text': 'another interesting application of', 'start': 1063.74, 'duration': 4.74}
{'text': 'incorporating symmetries can be found', 'start': 1065.96, 'duration': 4.38}
{'text': 'for DNA data', 'start': 1068.48, 'duration': 4.439}
{'text': 'these double helix sequences have a', 'start': 1070.34, 'duration': 4.8}
{'text': 'special inherent symmetry called reverse', 'start': 1072.919, 'duration': 4.5}
{'text': 'complement that can be taken into', 'start': 1075.14, 'duration': 5.039}
{'text': 'account in the model architecture', 'start': 1077.419, 'duration': 5.161}
{'text': 'finally variants of graph neural', 'start': 1080.179, 'duration': 4.801}
{'text': 'networks also incorporate interesting', 'start': 1082.58, 'duration': 4.08}
{'text': 'equivariances', 'start': 1084.98, 'duration': 4.5}
{'text': 'naturally all gnns are a permutation', 'start': 1086.66, 'duration': 5.58}
{'text': "active variant but it's also possible to", 'start': 1089.48, 'duration': 5.819}
{'text': 'extend them in a 3D setting to be for', 'start': 1092.24, 'duration': 6.059}
{'text': 'example rotation equivalent as well', 'start': 1095.299, 'duration': 4.981}
{'text': 'so as you can see there are plenty of', 'start': 1098.299, 'duration': 4.081}
{'text': 'interesting architectures or layers for', 'start': 1100.28, 'duration': 4.32}
{'text': 'different symmetry groups and the goal', 'start': 1102.38, 'duration': 4.14}
{'text': 'of the next video is to get a deeper', 'start': 1104.6, 'duration': 3.3}
{'text': 'understanding of some of the most', 'start': 1106.52, 'duration': 4.019}
{'text': "popular models out there that's it for", 'start': 1107.9, 'duration': 4.74}
{'text': 'this introduction part at this point you', 'start': 1110.539, 'duration': 4.02}
{'text': 'should be familiar with what groups are', 'start': 1112.64, 'duration': 5.159}
{'text': 'what Kaylee tables can be used for plus', 'start': 1114.559, 'duration': 5.641}
{'text': 'some of the group names and finally', 'start': 1117.799, 'duration': 4.681}
{'text': 'applications of equivariant neural', 'start': 1120.2, 'duration': 4.74}
{'text': "networks thanks for watching and I'll", 'start': 1122.48, 'duration': 5.12}
{'text': 'see you again in the next part', 'start': 1124.94, 'duration': 5.84}
{'text': 'thank you', 'start': 1127.6, 'duration': 3.18}

~~~

#### Exercise
주관성이 높은 영상과 객관성이 높은 영상을 임의적으로 선택하고 이에 대한 주관성/객관성 분석을 실시하여라. 

만약 가능하다면, 많은 영상을 보유한 특정 키워드가 주관성/객관성 혹은 긍정/부정에 영향을 받는 것이 있는지 가정하고 실시하시오.

#### Consideration
구어체인경우, 대다수가 완전한 문장이 아니고, 시간의 단위가 매우 짧다. 이를 어떻게 해결할지에 대해 고민하시오.

기본적으로는, 우리는 5분단위로 취합할수도 있으며, 혹은 300 words 내외 기준으로 취합할 수 도 있다.

고급파트로는, 우리는 완전한 문장의 집합 혹은 단락으로 구분할수 도 있다. 또는 정보 이론(shannon entropy)를 활용하여 놀라운 사실임을 판단하여 구분할 수 도 있다.

#### Consideration: Advanced

좀 더 구체적으로, 섀논 엔트로피에 대해 설명하면

가능한 결과 x1, ... ,xN을 가진 이산 확률 변수 X와 가능한 결과의 발생 확률은 P(x1), ..., p(xN) 라고 주어졌을대, 우리는 섀넌 엔트로피(Shannon Entropy)를 다음과 같이 구할 수 있다.
$H(X)=-Σ^{n}_{i=1}P(x_i) log P(x_i)$

즉, 일련의 자막(단어의 시퀀스)가 주어졌다면, 우리는 각 단어별로 등장한 것에 따라 확률을 구할 수 있다. 그런다음 각 확률 값을 Shannon Entropy에 대입하여 결과를 얻을 수있다. 그게 H(X) Entropy 값이다. 불확실성(즉 이전에 없는 새로운 결과라면)이 높으면 값은 매우 커진다. 다시 말해서, 새로운 단어가 자주 등장했다면 새로운 부분이라고도 판단할 수 있다.

python itertools를 이용하여 진행해보시기 바랍니다

# 요 약
YouTube API뿐만 아니라, Panopto, Brightcove, Vimeo, IBM Watson Media 등 다양한 비디오 플랫폼이 존재한다. 본 튜토리얼은 영상이 아닌 영상에 관련된 메타데이터(숫자, 텍스트)를 어떻게 분석해야하는지에 다루었다.


# 나아갈 방향
- 유튜브의 특정 카테고리의 특징을 반영한 서비스를 개발해 본다.
   - 왜 해당 서비스를 해야하는가? 
   - 단순 키워드에만 의존하는 것이 아닌가?
- 음성적인 부분과, 시각적인 부분을 함께 고려한 서비스를 개발해 본다. 
- 다양한 주제로 한 서비스를 모색해 본다.


*** 빅데이터 분석은 하나의 컴포넌트가 아닌, 여러개의 컴포넌트로 구성되는 파이프라인을 잘 설계하는 것이 큰 중요한 부분임을 알고 있어야 한다.

## 몇가지 아이디어

유튜브 상품 리뷰,
실시간 쇼핑/라이브 방송에 대한 반응성/극성 체크
