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
    - 原來寫code上出現有點問題,在send email的部分,一開始先在特定的問題關鍵字執行後Bot先回答並要求"receipient","subject","body",問了大略的問題後ai agent就真的開始進行邏輯推理,完成後就可以成功寄出去,且內容頗為完整的。
- 0429
    - 用GPU當Final Answer以後並沒有馬上而是直接重新尋找observation,而且重點是,**下一輪的thought竟然就卡住當機**
    - 目前關鍵看來是**return_intermediate_steps**,來仔細研究測試
        - 但其實也不是關鍵,他只是讓你來測試而已,你覺得用open_calculator可以總是成功執行是因為用了**return-direct=True**
- 0430
    - 用**langgraph**替代**langchain**
        - 目前可以先參考```Agentic_AI_Tool_test5.py```
    - 但現在的問題出在,我在輸入問題的時候,有時候需要找到關鍵字才能再進行後續的resoning,action...,但就像查詢天氣的時候,很容易找不到關鍵字(ex:城市)輸入到API,這是一個相對來說問題所在。
- 0501
    - 把```workflow_multi.py```跟```console_chatbot_langgraph_v2```分開避免搞混
- 0502    
    - 多設定給一般的cuda的選擇,在```workflow_multi_cuda.py```這個部分,可以回家測試用有cuda的狀況下是否可執行
    - 發現雖然我是從**tool_call**進到切入口,但不管怎麼樣他好像一定會執行到**agent**，所以才會出現
        - 建議還是用agent為主,除非真的是找不到,才會往**tool_call**走
- 0503
    - 考慮怎麼用Agentic AI 做出一個word
        - 結果先是從股票和財報下手

- 0505
    - 在**workflow.add_conditional_edges**跟**workflow.set_entry_point**兩者並不衝突,set_entry_point是在工作流程開始的點是agent,而add_conditional_edges則是留到tool_call時會要結束?還是繼續往回agent重新再做一次
    - 目前我在財報這個部分做一個簡單的Agentic AI,但我只是問公司的狀況,他只能給了最後4個季報和YOY的結果,要再思考還能做些甚麼其他的部分
- 0507
    - 要加入上下文歷史紀錄
        - 結果今天還未成功
    - 但我把agent tool跟人機問答分離出來,如果有問到某些tool的問題範圍之內,他就會執行此tool並執行,不然就利用我們的function ```simple_chatbot_response``` step by step response.
        - 想辦法回應後就不要再進入```call_tools_with_feedback``` => 在message最後面多加個SystemMessage(context=[FINISH RESPONSE])
- 0508
    - 要加入上下文歷史紀錄
    - 給一個另一個功能: 進入會議模式 1.電腦亮度變暗 2.檢測網路狀況 3.
- 0509
    - 之前的code```workflow_multi_history```他的定義太嚴苛了,我改用```workflow_multi_fuzzy_tool.py```讓問題可以比較有空間,能幫你做tool,也可以幫你處理一般問答
    - 先做一個檢測網路狀況的tool,叫做**Checking_Internet.exe** => 但是把它compile成.exe卻一直失敗
    - 現在來玩另一個,就是錄音錄影的功能,先進行按"q"就會結束並儲存的錄音錄影tool ```Open_Camera_Microphone.exe```
    - 可以考慮用另一個tool來關閉```Open_Camera_Microphone.exe```,Github Copilot給的是用文字出現就跳出.exe,但我還要想一想
- 0512
    - 希望利用```Close_Camera_Microphone.exe```來關閉之前的```Open_Camera_Microphone.exe```
        - 暫時不考慮 ```Close_Camera_Microphone.exe```來關閉,用手動關閉吧
    - 先在```Open_Camera_Microphone.exe```裡面產生一個filename時也利用Agentic AI來處理成一個更完整的.mp4 filename => 成功!!
    - 我回去思考,在input的時候我是給了prompt然後LLM進行Agentic AI並給然後再observation,thought,reasoning, react...,但是在回應我都給一個制式的回應,如果成功救回應:xxxxx,如果失敗就回應:ooooo,必須重新思考,在回應的部分也可以使用Agentic AI?
        - 就是在結束後要response時給了response_prompt....然後```tool_result = llm._call(response_prompt).strip()```,他就是回應Agentic AI的自動產生的
        - 注意在response_prompt的部分回應主軸就好,無須多寫其他的(without any additional text.")
        - 基本上我就是讓agentic ai依照observation,reasoning step, action一直找到他覺得OK以後把最後一個部分(brightness,master-volume,app-volume)的int value回傳給.exe 去執行,並回應給user,當然時間就比較久
    - 明天還是盡力用history來去找之前的問答然後重新修改不管是brightness還是audio volume的大小
- 0513
    - 先處理一般的問答,注意,有時候如果問題是一般的回應,但卻進入action的執行狀況,請用tool.name == "Chatbot"
    - 在一般的問答(chatbot_tool),我直接把prompt拿掉,讓回答單純很多,不用一直進行自問自答
    - 在prompt的部分如果利用Agentic AI讓LLM自己邏輯推理,prompt就要多寫一些
    - Thu.的時要測試"Activate the meeting mode and I want the brightness value is less than 30" => 可以在console_chatbot_langgraph_v4_0513.py下執行
    - 我在responseb如果是利用agentic ai時,他就不會有"[TOOL_EXECUTED]",而在python <main.py>就會出現"🤖 {msg.content}",要特別注意
- 0514
    - "I want to activate the meeting mode and the brightness is less than 30"最新的測試也成功
    - 發現在第一個問題OK,但接續的問題時會出現 **Failed to parse meeting mode parameters: {str(e)}. Please ensure the input format is correct.** 
        - 主要問題是因為在meeting_mode_tool裡面她要確認brightness,master_value,app_value如果沒有輸入的話她就會出現error,所以可以在外面先把這三個都定義到global varialbe,這樣就解決問題
    - 第一個問題讓brightness變小,但第二個問題內容我有說"but keep the value of brightness"但brightness還是變大了,這該怎麼解決?
        - 在prompt中增加```f"Here is the chat history for reference: {memory.chat_memory.messages}. "```(chat_history)
        - # 更新全局上下文
            meeting_mode_context["brightness"] = brightness
            meeting_mode_context["master_volume"] = master_volume
            meeting_mode_context["app_volume"] = app_volume
        - 可以用以下的問答來成功做出我的Demo
            - Activate the meeting mode and I want the brightness value is less than 30
            - The master_value is too small, I want the value of it can upgrade but keep the value of brightness the same
    - 接下來我把meeting mode打掉,取代的是連續做三個tool(brightness,aduio_volume,power server)
    