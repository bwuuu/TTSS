import streamlit as st
import requests
import json
from datetime import datetime

# Page config
st.set_page_config(
    page_title="AI CrewAI Workspace",
    page_icon="🕵️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .agent-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'conversations' not in st.session_state:
    st.session_state.conversations = []

# Agents data
AGENTS = {
    "tinker": {
        "name": "Tinker",
        "role": "Technical Innovator",
        "description": "A brilliant engineer who loves to build, fix, and optimize systems.",
        "emoji": "🔧",
        "specialties": ["Code Generation", "System Architecture", "Problem Solving"]
    },
    "tailor": {
        "name": "Tailor",
        "role": "Content Creator", 
        "description": "A meticulous wordsmith who crafts perfect content and documentation.",
        "emoji": "✂️",
        "specialties": ["Content Writing", "Documentation", "Communication"]
    },
    "soldier": {
        "name": "Soldier",
        "role": "Executor",
        "description": "A disciplined professional who gets things done efficiently.",
        "emoji": "⚔️",
        "specialties": ["Project Management", "Execution", "Quality Control"]
    },
    "spy": {
        "name": "Spy",
        "role": "Intelligence Analyst",
        "description": "A perceptive analyst who gathers information and provides insights.",
        "emoji": "🕵️",
        "specialties": ["Research", "Analysis", "Strategic Planning"]
    },
    "mr_smiley": {
        "name": "Mr. Smiley",
        "role": "CEO",
        "description": "A charismatic leader who coordinates the team and makes decisions.",
        "emoji": "😊",
        "specialties": ["Leadership", "Strategy", "Coordination"]
    }
}

def query_huggingface(prompt, api_token, model="microsoft/DialoGPT-medium"):
    """Query Hugging Face API"""
    if not api_token:
        return "⚠️ Please add your Hugging Face API token in the sidebar."
    
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {api_token}"}
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": 150,
            "temperature": 0.7,
            "do_sample": True
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "")
                # Clean up the response
                if prompt in generated_text:
                    generated_text = generated_text.replace(prompt, "").strip()
                return generated_text if generated_text else "I'm thinking about your question..."
            return "Let me process that for you..."
        else:
            return f"API Error: {response.status_code}. Please check your token."
    except Exception as e:
        return f"Connection error: {str(e)}"

def create_agent_prompt(agent_key, user_input):
    """Create a prompt for the specific agent"""
    agent = AGENTS[agent_key]
    
    return f"""You are {agent['name']}, a {agent['role']}. {agent['description']}

Your specialties: {', '.join(agent['specialties'])}

User question: {user_input}

Respond as {agent['name']} in a helpful, professional way (max 100 words):"""

# Header
st.markdown("""
<div class="main-header">
    <h1>🕵️ AI CrewAI Workspace</h1>
    <p>Meet your team of AI agents: Tinker, Tailor, Soldier, Spy & Mr. Smiley</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    api_token = st.text_input(
        "Hugging Face API Token",
        type="password",
        help="Get free token from huggingface.co/settings/tokens"
    )
    
    model = st.selectbox(
        "AI Model",
        ["microsoft/DialoGPT-medium", "gpt2", "distilgpt2"],
        help="Choose your AI model"
    )
    
    st.divider()
    
    st.header("📊 Stats")
    st.metric("Total Conversations", len(st.session_state.conversations))
    
    if st.button("Clear History"):
        st.session_state.conversations = []
        st.rerun()

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Chat with Your Agents")
    
    # Agent selector
    selected_agent = st.selectbox(
        "Choose your agent:",
        options=list(AGENTS.keys()),
        format_func=lambda x: f"{AGENTS[x]['emoji']} {AGENTS[x]['name']} - {AGENTS[x]['role']}"
    )
    
    agent = AGENTS[selected_agent]
    
    # Agent info
    st.markdown(f"""
    <div class="agent-card">
        <h3>{agent['emoji']} {agent['name']}</h3>
        <p><strong>Role:</strong> {agent['role']}</p>
        <p><strong>Description:</strong> {agent['description']}</p>
        <p><strong>Specialties:</strong> {', '.join(agent['specialties'])}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat input
    user_input = st.text_area(
        "Your message:",
        placeholder=f"Ask {agent['name']} about their expertise...",
        height=100
    )
    
    if st.button("Send Message", type="primary"):
        if user_input.strip():
            with st.spinner(f"{agent['name']} is thinking..."):
                prompt = create_agent_prompt(selected_agent, user_input)
                response = query_huggingface(prompt, api_token, model)
                
                # Save conversation
                st.session_state.conversations.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "agent": agent['name'],
                    "agent_emoji": agent['emoji'],
                    "user_input": user_input,
                    "response": response
                })
                
                st.rerun()
    
    # Chat history
    st.subheader("💭 Recent Conversations")
    
    if st.session_state.conversations:
        # Show last 5 conversations
        for conv in reversed(st.session_state.conversations[-5:]):
            with st.expander(f"{conv['agent_emoji']} {conv['agent']} - {conv['timestamp']}"):
                st.markdown(f"**You:** {conv['user_input']}")
                st.markdown(f"**{conv['agent']}:** {conv['response']}")
    else:
        st.info("No conversations yet. Start chatting with an agent!")

with col2:
    st.header("👥 Your AI Team")
    
    # Show all agents
    for key, agent in AGENTS.items():
        # Count conversations with this agent
        agent_convs = len([c for c in st.session_state.conversations if c['agent'] == agent['name']])
        status = "🟢 Active" if agent_convs > 0 else "⚪ Ready"
        
        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border: 1px solid #ddd;">
            <h4>{agent['emoji']} {agent['name']}</h4>
            <p style="margin: 0.5rem 0; color: #666;">{agent['role']}</p>
            <small>{status} • {agent_convs} chats</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Quick actions
    st.header("🚀 Quick Start")
    
    if st.button("💡 Ask Tinker about coding"):
        st.session_state.quick_message = "Help me write a Python function"
        
    if st.button("📝 Ask Tailor for content"):
        st.session_state.quick_message = "Write a professional email"
        
    if st.button("⚔️ Ask Soldier for planning"):
        st.session_state.quick_message = "Help me organize a project"

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🚀 <strong>AI CrewAI Workspace</strong> • Built with Streamlit • Powered by Hugging Face</p>
    <p>Mobile-friendly • Session memory • Ready for Railway deployment</p>
</div>
""", unsafe_allow_html=True)
