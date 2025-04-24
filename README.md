# Agentic AI Send Email


- 0419
    - 用gmail寄信在NTU的wifi是可行的
- 0421
    - 但在公司用gmail寄信到另一個gmail卻變成不可行,原因不明. => IT會擋住寄信的狀況
    - 上午嘗試用respond的時候希望可以streaming token,但後來想想再做Agentic AI的時候除了真的最後輸出以外,他也必須經過很多的過程,如果你是一個token一個token的吐出來,那過程中很容易就出錯了 =>目前看來就暫時不去思考取代用streaming token了
    - 有時候我試圖打開tool成功,但最後的回答卻出現**Sorry, I couldn't process your request. Error: Exception from src\inference\src\cpp\infer_request.cpp:245:
Check '*roi_begin <= *max_dim' failed at src\inference\src\dev\make_tensor.cpp:33**,很可能是因為雖然已經成功之行,但後續的observation,thougt,final answer還是繼續運作,直到最後才告知結果,難道沒辦法只要執行成功後就跳出結果,之後就不要繼續往下走?
        - 可能**description**是非常關鍵的點,他可以決定當你開啟後就停止之後的動作?
    - **ZERO_SHOT_REACT_DESCRIPTION** 這個是給沒有參數的function,有參數的function是回出現error的
    - 我相信**description**就是能agentic ai要不要繼續run還是停止的點
- 0422
    - 今天試圖打開計算機"Open Calculator"卻又失敗,不太知道問題出在哪?
- 0424
    - 我在**console_chatbot.py**裡面有設定```if "Stopping further processing" in response: pass```,結果是出現正確的回應和打開calculator
    - 但那只是更侷限,一般來說,我還是用```Open the calculator```就可以成功打開計算機,但也帶來問題**是不是問題要非常的詳細才能讓AI inference出正確的答案?**
    - 多加了天氣的問題(Get Weather)來看是否找到答案
        - 發現有時候雖然已經找到適合的observation和action和thogut,可是AI會繼續推理(inference)不停止,導致最後類似segmentation fault的error(Thought:Bot: Sorry, I couldn't process your request. Error: Exception from src\inference\src\cpp\infer_request.cpp:245:
Check '*roi_begin <= *max_dim' failed at src\inference\src\dev\make_tensor.cpp:33)

