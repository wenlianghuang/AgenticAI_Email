from langgraph.graph import StateGraph
from typing import TypedDict

# 定義狀態結構
class MyState(TypedDict):
    input: str
    decision: str

# 定義節點 A，接收輸入並傳遞到 A_1
def node_a(state: MyState) -> MyState:
    user_input = input("請輸入內容 (e.g., 'I want apple'): ")
    print(f"A 接收到輸入: {user_input}")
    return {"input": user_input, "decision": ""}

# 定義節點 A_1，根據輸入決定 "decision"
def node_a_1(state: MyState) -> MyState:
    user_input = state["input"]
    if "apple" in user_input.lower():
        decision = "to_b"
    else:
        decision = "to_c"
    print(f"A_1 判斷輸入為: {user_input} -> 決定走向 {decision}")
    return {"input": user_input, "decision": decision}

# 節點 B
def node_b(state: MyState) -> MyState:
    print("執行 B：你提到了蘋果 🍎")
    return state

# 節點 C
def node_c(state: MyState) -> MyState:
    print("執行 C：你沒提到蘋果 🤔")
    return state

# 初始化 StateGraph
graph = StateGraph(MyState)

# 加入節點
graph.add_node("A", node_a)
graph.add_node("A_1", node_a_1)
graph.add_node("B", node_b)
graph.add_node("C", node_c)

# 設定起點
graph.set_entry_point("A")

# A 直接跳轉到 A_1
graph.add_edge("A", "A_1")

# 根據 A_1 的回傳值中的 'decision' 做條件跳轉
graph.add_conditional_edges(
    "A_1",
    lambda state: state["decision"],  # 根據 state["decision"] 的值決定跳哪個 node
    {
        "to_b": "B",
        "to_c": "C"
    }
)

# 設定 B 和 C 為終點
graph.set_finish_point("B")
graph.set_finish_point("C")

# 編譯 graph
app = graph.compile()

# 執行應用程式
print("\n🧪 啟動應用程式")
app.invoke({"input": ""})  # 初始輸入為空，實際輸入由 node_a 處理