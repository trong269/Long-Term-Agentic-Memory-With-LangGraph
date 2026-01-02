# Long-Term Agentic Memory With LangGraph

Khóa học xây dựng Agent có bộ nhớ dài hạn sử dụng LangGraph - Hợp tác giữa DeepLearning.AI và LangChain.

## 📚 Giới thiệu

Khóa học này hướng dẫn cách xây dựng một **Email Assistant Agent** thông minh với khả năng nhớ dài hạn. Agent có thể tự động phân loại email, soạn thảo phản hồi, lên lịch cuộc họp và học hỏi từ trải nghiệm trước đó.

### Các loại bộ nhớ được áp dụng:
- **Semantic Memory (Bộ nhớ ngữ nghĩa)**: Lưu trữ thông tin về người dùng và các sự kiện
- **Episodic Memory (Bộ nhớ tình huống)**: Học từ các ví dụ xử lý email trước đó
- **Procedural Memory (Bộ nhớ thủ tục)**: Tối ưu hóa quy trình làm việc qua phản hồi

---

## 📖 Nội dung từng Lesson

### Lesson 2: Baseline Email Assistant
**Xây dựng trợ lý email cơ bản**

Tạo agent đơn giản với ba chức năng chính:
- **Triage (Phân loại)**: Chia email thành 3 loại - Ignore (Bỏ qua), Notify (Thông báo), Respond (Phản hồi)
- **Tools**: Viết email, lên lịch họp, kiểm tra lịch trống
- **Router**: Định tuyến email dựa trên phân loại

**Code minh họa:**
```python
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

# Định nghĩa Router để phân loại email
class Router(BaseModel):
    reasoning: str = Field(description="Lý do phân loại")
    classification: Literal["ignore", "respond", "notify"]

llm = init_chat_model("openai:gpt-4o-mini")
llm_router = llm.with_structured_output(Router)

# Tạo các công cụ cho agent
@tool
def write_email(to: str, subject: str, content: str):
    return f"Email sent to {to}"

@tool  
def schedule_meeting(attendees: list, subject: str):
    return f"Meeting scheduled: {subject}"
```

---

### Lesson 3: Email Assistant với Semantic Memory
**Thêm bộ nhớ ngữ nghĩa để ghi nhớ thông tin người dùng**

Nâng cấp agent với khả năng:
- **Lưu trữ thông tin**: Ghi nhớ các sự kiện về người dùng vào memory store
- **Tìm kiếm thông tin**: Truy xuất thông tin liên quan từ bộ nhớ
- **Sử dụng context**: Áp dụng thông tin đã lưu vào phản hồi

**Code minh họa:**
```python
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool

# Khởi tạo memory store
store = InMemoryStore(
    index={"embed": "openai:text-embedding-3-small"}
)

# Tạo tools quản lý bộ nhớ
manage_memory_tool = create_manage_memory_tool(
    namespace=("email_assistant", "{user_id}", "collection")
)

search_memory_tool = create_search_memory_tool(
    namespace=("email_assistant", "{user_id}", "collection")
)

# Agent có thể lưu và tìm kiếm thông tin
tools = [write_email, schedule_meeting, 
         manage_memory_tool, search_memory_tool]
```

---

### Lesson 4: Thêm Episodic Memory
**Học từ ví dụ và phản hồi của người dùng**

Cải thiện khả năng phân loại với:
- **Few-shot examples**: Lưu trữ các ví dụ phân loại email từ người dùng
- **Human-in-the-loop**: Thu thập phản hồi từ người dùng
- **Retrieval**: Tìm kiếm ví dụ tương tự để cải thiện quyết định

**Code minh họa:**
```python
# Template để format ví dụ few-shot
template = """
Email Subject: {subject}
Email From: {from_email}
> Triage Result: {result}
"""

# Hàm triage với episodic memory
def triage_router(state, config, store):
    # Tìm kiếm ví dụ tương tự
    namespace = ("email_assistant", config['user_id'], "examples")
    examples = store.search(
        namespace, 
        query=str({"email": state['email_input']})
    )
    
    # Format examples cho prompt
    formatted_examples = format_few_shot_examples(examples)
    
    # Phân loại với examples
    system_prompt = triage_prompt.format(
        examples=formatted_examples,
        **profile
    )
    result = llm_router.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])
    
    return result.classification
```

---

### Lesson 5: Thêm Procedural Memory
**Tối ưu hóa instructions dựa trên feedback - Bộ nhớ thủ tục**

Procedural Memory là khả năng agent tự động cải thiện **cách thức hoạt động** qua thời gian bằng cách cập nhật system prompts dựa trên phản hồi của người dùng.

#### 🎯 Khái niệm chính

**Procedural Memory** lưu trữ "know-how" - cách thực hiện công việc:
- Không phải lưu **sự kiện** (Semantic Memory)
- Không phải lưu **ví dụ** (Episodic Memory)  
- Mà lưu **quy trình và hướng dẫn** cần cải thiện

#### 📋 Quy trình 3 bước

**Bước 1: Lưu trữ instructions động trong Store**

Agent không dùng prompts cố định mà lấy từ memory store:

```python
def triage_router(state, config, store):
    user_id = config['configurable']['langgraph_user_id']
    namespace = (user_id,)
    
    # Lấy hoặc khởi tạo triage_ignore instructions
    result = store.get(namespace, "triage_ignore")
    if result is None:
        # Lần đầu: lưu instructions mặc định
        store.put(namespace, "triage_ignore", 
                 {"prompt": "Marketing newsletters, spam emails..."})
        ignore_prompt = "Marketing newsletters, spam emails..."
    else:
        # Lần sau: dùng instructions đã được tối ưu
        ignore_prompt = result.value['prompt']
    
    # Tương tự cho triage_notify và triage_respond
    result = store.get(namespace, "triage_notify")
    notify_prompt = result.value['prompt'] if result else default_notify
    
    result = store.get(namespace, "triage_respond")  
    respond_prompt = result.value['prompt'] if result else default_respond
    
    # Sử dụng instructions động
    system_prompt = triage_system_prompt.format(
        triage_no=ignore_prompt,
        triage_notify=notify_prompt,
        triage_email=respond_prompt
    )
```

**Bước 2: Thu thập feedback và conversation history**

Sau khi agent xử lý email, lưu lại conversation và feedback:

```python
# Chạy agent
response = email_agent.invoke(
    {"email_input": email_input},
    config=config
)

# Tạo training data với feedback
conversations = [
    (
        response['messages'],  # Lịch sử conversation
        "Always sign your emails `John Doe`"  # Feedback từ user
    )
]
```

**Bước 3: Sử dụng LLM Optimizer để cập nhật prompts**

Dùng `create_multi_prompt_optimizer` để tự động cải thiện prompts:

```python
from langmem import create_multi_prompt_optimizer

# Định nghĩa các prompts cần tối ưu
prompts = [
    {
        "name": "main_agent",
        "prompt": store.get(("lance",), "agent_instructions").value['prompt'],
        "update_instructions": "Keep instructions short and to the point",
        "when_to_update": "Update when feedback on writing emails or scheduling"
    },
    {
        "name": "triage-ignore", 
        "prompt": store.get(("lance",), "triage_ignore").value['prompt'],
        "update_instructions": "Keep instructions short and to the point",
        "when_to_update": "Update when feedback on which emails to ignore"
    },
    {
        "name": "triage-notify",
        "prompt": store.get(("lance",), "triage_notify").value['prompt'],
        "update_instructions": "Keep instructions short and to the point",
        "when_to_update": "Update when feedback on notification emails"
    },
    {
        "name": "triage-respond",
        "prompt": store.get(("lance",), "triage_respond").value['prompt'],
        "update_instructions": "Keep instructions short and to the point",
        "when_to_update": "Update when feedback on which emails need response"
    }
]

# Tạo optimizer
optimizer = create_multi_prompt_optimizer(
    "anthropic:claude-3-5-sonnet-latest",
    kind="prompt_memory"
)

# Chạy optimization
updated_prompts = optimizer.invoke({
    "trajectories": conversations,  # Lịch sử + feedback
    "prompts": prompts              # Prompts hiện tại
})

# Lưu prompts đã được tối ưu vào store
for i, updated_prompt in enumerate(updated_prompts):
    old_prompt = prompts[i]
    if updated_prompt['prompt'] != old_prompt['prompt']:
        name = old_prompt['name']
        print(f"✅ Updated {name}")
        
        if name == "main_agent":
            store.put(("lance",), "agent_instructions",
                     {"prompt": updated_prompt['prompt']})
        elif name == "triage-ignore":
            store.put(("lance",), "triage_ignore",
                     {"prompt": updated_prompt['prompt']})
        elif name == "triage-notify":
            store.put(("lance",), "triage_notify",
                     {"prompt": updated_prompt['prompt']})
        elif name == "triage-respond":
            store.put(("lance",), "triage_respond",
                     {"prompt": updated_prompt['prompt']})
```

#### 🔄 Vòng lặp cải tiến liên tục

```
1. Agent xử lý email với prompts hiện tại
        ↓
2. Thu thập feedback từ user
        ↓
3. LLM Optimizer phân tích và cải thiện prompts
        ↓
4. Lưu prompts mới vào store
        ↓
5. Lần sau agent dùng prompts đã được cải thiện
        ↓
   (Quay lại bước 1)
```

#### 💡 Ví dụ thực tế

**Trước khi có Procedural Memory:**
```
User: "Ignore emails from Alice Jones"
→ Agent vẫn phản hồi emails từ Alice Jones
```

**Sau khi áp dụng Procedural Memory:**
```python
# Feedback
conversations = [(response['messages'], "Ignore any emails from Alice Jones")]

# Optimizer cập nhật prompt
updated = optimizer.invoke({"trajectories": conversations, "prompts": prompts})

# Prompt "triage-ignore" được cập nhật tự động:
# Trước: "Marketing newsletters, spam emails..."
# Sau:  "Marketing newsletters, spam emails, emails from Alice Jones..."

# Lần sau agent sẽ tự động ignore emails từ Alice Jones
```

#### 🎓 Điểm mạnh của Procedural Memory

- ✅ **Tự động học**: Không cần manually cập nhật prompts
- ✅ **Cải thiện liên tục**: Agent ngày càng hiểu rõ preferences của user
- ✅ **Scalable**: Có thể tối ưu nhiều prompts cùng lúc
- ✅ **Personalized**: Mỗi user có bộ instructions riêng

#### ⚙️ Cấu hình quan trọng

```python
# Mỗi prompt cần 3 thông tin:
{
    "name": "tên_prompt",
    "prompt": "nội_dung_hiện_tại",
    "update_instructions": "Hướng dẫn cho optimizer",
    "when_to_update": "Điều kiện để cập nhật"
}

# Optimizer cần:
- Model: "anthropic:claude-3-5-sonnet-latest"  
- Kind: "prompt_memory"
- Trajectories: [(conversation, feedback)]
```

---

## 🚀 Cách sử dụng

1. **Cài đặt dependencies:**
```bash
pip install -r requirements.txt
```

2. **Thiết lập API keys:**
```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
```

3. **Chạy notebooks theo thứ tự:**
- `lesson2.ipynb` - Agent cơ bản
- `lesson_3.ipynb` - Thêm Semantic Memory
- `lesson_4.ipynb` - Thêm Episodic Memory  
- `lesson_5.ipynb` - Thêm Procedural Memory

---

## 🎯 Kết quả đạt được

Sau khóa học, bạn sẽ:
- ✅ Hiểu 3 loại bộ nhớ: Semantic, Episodic, Procedural
- ✅ Xây dựng agent với LangGraph có khả năng nhớ dài hạn
- ✅ Triển khai memory store và retrieval system
- ✅ Áp dụng human-in-the-loop để cải thiện agent
- ✅ Tối ưu hóa prompts dựa trên feedback

---

## 📦 Công nghệ sử dụng

- **LangGraph**: Framework xây dựng agent
- **LangChain**: Công cụ tích hợp LLM
- **OpenAI GPT-4**: Mô hình ngôn ngữ
- **Claude 3.5 Sonnet**: Mô hình Anthropic
- **InMemoryStore**: Hệ thống lưu trữ bộ nhớ

---

## 📝 License

Khóa học được cung cấp bởi **DeepLearning.AI** và **LangChain**.

---

*Tạo bởi Harrison Chase - Co-Founder & CEO của LangChain*