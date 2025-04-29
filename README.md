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
        - 也不是,但在ai agent的過程中,ai在打開計算機後他還是會繼續依照環境(observation)來找tool並action,但這可能就不是我決定了,是AI會來繼續判斷有沒有持續做下去的理由
    - 多加了天氣的問題(Get Weather)來看是否找到答案
        - 發現有時候雖然已經找到適合的observation和action和thogut,可是AI會繼續推理(inference)不停止,導致最後類似segmentation fault的error(Thought:Bot: Sorry, I couldn't process your request. Error: Exception from src\inference\src\cpp\infer_request.cpp:245:
Check '*roi_begin <= *max_dim' failed at src\inference\src\dev\make_tensor.cpp:33)

- 0425
    - 必須對**return_intermediate_steps**有詳細的了解
        - 代理會返回執行過程中的中間步驟（intermediate steps）。
這些中間步驟包括代理的推理過程、選擇的工具、工具的輸入和輸出等。
主要用於調試和分析代理的內部邏輯。

- 0426
    - 今天又嘗試想要send email, 收件者:xxx, Subject: AI Agent, Body: Try to send an email by ai agent => 經過邏輯推理後, Subject: AI Agent Email Sending Request, Body: The user is requesting assistance with sending an email using an AI agent
    - 但一樣的問題,在Finial Answer以後LLM還是覺得應該繼續在新的respond繼續邏輯推理,直到Failed to send email. Error: Exception from src\inference\src\cpp\infer_request.cpp:245:
Check '*roi_begin <= *max_dim' failed at src\inference\src\dev\make_tensor.cpp:33
    - 原來寫code上出現有點問題,在send email的部分,一開始先在特定的問題關鍵字執行後Bot先回答並要求"receipient","subject","body",問了大略的問題後ai agent就真的開始進行邏輯推理,完成後就可以成功寄出去,且內容鰻完整的。

