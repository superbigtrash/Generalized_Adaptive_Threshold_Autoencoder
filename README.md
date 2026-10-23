# Transformer AutoEncoder
 
### 2026년 제6회 K-인공지능 제조데이터 분석 경진대회

Our idea starts from LSTM_AutoEncoder given as guide book for beginner.

So we think how about using Transformer and AutoEncoder for this problem?

We use Transformer Encoder as Encoder and Decoder from AutoEncoder.


### KAMP NOTICE and Dataset

https://www.kamp-ai.kr/contestNoticeDetail?PAGE=1&CPT_NOTICE_SEQ=28&SEARCH_TXT=&LINE_PER_PAGE=10

### Final Results
We made generalized version of adaptive threshold autoencoder(using transformer / mamba) 

The results are below, predicted threshold(orange color)
                       input value(blue color)(normal data + abnormal data / upper, under vibration, current, equipment state)
<img width="1400" height="600" alt="reconstruction_error_1" src="https://github.com/user-attachments/assets/924b773a-f945-44ef-9718-ad285ad5302c" />
<img width="1400" height="600" alt="reconstruction_error_2" src="https://github.com/user-attachments/assets/3a8833a0-2710-401b-84e0-f4bb5771c354" />
<img width="700" height="700" alt="confusion_1" src="https://github.com/user-attachments/assets/5030b81d-bd5c-459d-91ae-c1f9152bc84f" />
<img width="700" height="827" alt="eval_1" src="https://github.com/user-attachments/assets/0122b71e-1833-4810-b404-96e079bcc9d2" />

This problem is done at 2026.10.07



