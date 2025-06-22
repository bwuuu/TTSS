import streamlit as st
import os
from datetime import datetime
import json
from typing import Dict, List, Any
import requests
from dataclasses import dataclass
from enum import Enum

# Page configuration
st.set_page_config(
    page_title="AI Workspace - CrewAI Hub",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for mobile responsiveness and styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .agent-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .agent-card h4 {
        color: #007bff;
        margin-bottom: 0.5rem;
    }
    
    .status-active {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-idle {
        color: #6c757d;
    }
    
    .memory-box {
        background: #e9ecef;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-family: monospace;
        font-size: 0.9rem;
    }
    
    @media (max-width: 768px) {
        .main-header {
            padding: 0.5rem;
        }
        
        .agent-card {
            padding: 1rem;
            margin: 0.5rem 0;
        }
    }
</style>
""", unsafe_allow_html=True)

class AgentRole(Enum):
    TINKER = "tinker"
    TAILOR = "tailor"
    SOLDIER = "soldier"
    SPY = "spy"
    CEO = "ceo"

@dataclass
class Agent:
    name: str
    role: AgentRole
    description: str
    persona: str
    specialties: List[str]
    status: str = "idle"

class HuggingFaceAPI:
    def __init__(self, api_token: str = None):
        self.api_token = api_token or os.getenv("HUGGINGFACE_API_TOKEN")
        self.base_url = "https://api-inference.huggingface.co/models"
        self.headers = {"Authorization": f"Bearer {self.api_token}"} if self.api_token else {}
    
    def query_model(self, model_name: str, prompt: str, max_length: int = 150) -> str:
        """Query Hugging Face Inference API"""
        if not self.api_token:
            return "⚠️ Hugging Face API token not configured. Please add your token to continue."
        
        url = f"{self.base_url}/{model_name}"
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_length": max_length,
                "temperature": 0.7,
                "do_sample": True
            }
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "No response generated")
                return str(result)
            else:
                return f"API Error: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Error: {str(e)}"

class SessionMemory:
    """Simple session-based memory system"""
    
    @staticmethod
    def initialize():
        if 'session_memory' not in st.session_state:
            st.session_state.session_memory = {
                'conversations': [],
                'agent_states': {},
                'project_context': {},
                'created_at': datetime.now().isoformat()
            }
    
    @staticmethod
    def add_conversation(agent: str, user_input: str, response: str):
        st.session_state.session_memory['conversations'].append({
            'timestamp': datetime.now().isoformat(),
            'agent': agent,
            'user_input': user_input,
            'response': response
        })
    
    @staticmethod
    def get_conversation_history(agent: str = None, limit: int = 10):
        conversations = st.session_state.session_memory['conversations']
        if agent:
            conversations = [c for c in conversations if c['agent'] == agent]
        return conversations[-limit:]
    
    @staticmethod
    def update_agent_state(agent: str, state: Dict):
        st.session_state.session_memory['agent_states'][agent] = state
    
    @staticmethod
    def clear_memory():
        st.session_state.session_memory = {
            'conversations': [],
            'agent_states': {},
            'project_context': {},
            'created_at': datetime.now().isoformat()
        }

class CrewAIWorkspace:
    def __init__(self):
        self.hf_api = HuggingFaceAPI()
        self.agents = self._initialize_agents()
        SessionMemory.initialize()
    
    def _initialize_agents(self) -> Dict[str, Agent]:
        return {
            "tinker": Agent(
                name="Tinker",
                role=AgentRole.TINKER,
                description="The Technical Innovator",
                persona="A brilliant engineer who loves to build, fix, and optimize systems. Always curious about how things work.",
                specialties=["Code Generation", "System Architecture", "Problem Solving", "Technical Innovation"]
            ),
            "tailor": Agent(
                name="Tailor",
                role=AgentRole.TAILOR,
                description="The Content Creator",
                persona="A meticulous wordsmith who crafts perfect content, adapts messaging, and ensures quality.",
                specialties=["Content Writing", "Documentation", "Communication", "Quality Assurance"]
            ),
            "soldier": Agent(
                name="Soldier",
                role=AgentRole.SOLDIER,
                description="The Executor",
                persona="A disciplined professional who gets things done efficiently and follows through on commitments.",
                specialties=["Project Management", "Execution", "Process Optimization", "Quality Control"]
            ),
            "spy": Agent(
                name="Spy",
                role=AgentRole.SPY,
                description="The Intelligence Analyst",
                persona="A perceptive analyst who gathers information, identifies patterns, and provides strategic insights.",
                specialties=["Research", "Analysis", "Intelligence Gathering", "Strategic Planning"]
            ),
            "mr_smiley": Agent(
                name="Mr. Smiley",
                role=AgentRole.CEO,
                description="The CEO Persona",
                persona="A charismatic leader who coordinates the team, makes strategic decisions, and ensures success.",
                specialties=["Leadership", "Strategy", "Coordination", "Decision Making"]
            )
        }
    
    def get_agent_prompt(self, agent_key: str, user_input: str) -> str:
        agent = self.agents[agent_key]
        context = SessionMemory.get_conversation_history(agent_key, limit=3)
        
        context_str = ""
        if context:
            context_str = "\n\nRecent conversation context:\n"
            for conv in context[-2:]:  # Last 2 exchanges
                context_str += f"User: {conv['user_input']}\nYou: {conv['response']}\n"
        
        return f"""You are {agent.name}, {agent.description}.

Your persona: {agent.persona}

Your specialties: {', '.join(agent.specialties)}

Instructions: Respond as {agent.name} would, staying in character. Be helpful, professional, and leverage your specialties. Keep responses concise but informative.

{context_str}

Current user request: {user_input}

Response:"""

def main():
    # Initialize workspace
    workspace = CrewAIWorkspace()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🕵️ AI CrewAI Workspace</h1>
        <p>Your intelligent team of AI agents at your service</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Configuration
        api_token = st.text_input(
            "Hugging Face API Token",
            type="password",
            help="Get your free token from huggingface.co/settings/tokens"
        )
        
        if api_token:
            workspace.hf_api.api_token = api_token
            workspace.hf_api.headers = {"Authorization": f"Bearer {api_token}"}
        
        model_name = st.selectbox(
            "Select Model",
            [
                "microsoft/DialoGPT-medium",
                "gpt2",
                "distilgpt2",
                "facebook/blenderbot-400M-distill"
            ],
            help="Choose your preferred language model"
        )
        
        st.divider()
        
        # Memory Management
        st.header("🧠 Session Memory")
        
        memory_stats = st.session_state.get('session_memory', {})
        if memory_stats:
            st.metric("Conversations", len(memory_stats.get('conversations', [])))
            st.metric("Active Agents", len(memory_stats.get('agent_states', {})))
        
        if st.button("Clear Memory", type="secondary"):
            SessionMemory.clear_memory()
            st.success("Memory cleared!")
            st.rerun()
        
        # Export session data
        if st.button("Export Session"):
            data = json.dumps(st.session_state.session_memory, indent=2)
            st.download_button(
                "Download Session Data",
                data,
                f"session_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json"
            )
    
    # Main workspace area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎯 Agent Interaction")
        
        # Agent selection
        selected_agent = st.selectbox(
            "Choose your agent:",
            options=list(workspace.agents.keys()),
            format_func=lambda x: f"{workspace.agents[x].name} - {workspace.agents[x].description}",
            key="agent_selector"
        )
        
        agent = workspace.agents[selected_agent]
        
        # Display agent info
        st.markdown(f"""
        <div class="agent-card">
            <h4>{agent.name}</h4>
            <p><strong>Role:</strong> {agent.description}</p>
            <p><strong>Persona:</strong> {agent.persona}</p>
            <p><strong>Specialties:</strong> {', '.join(agent.specialties)}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Chat interface
        user_input = st.text_area(
            "Your message:",
            placeholder=f"Ask {agent.name} something related to their expertise...",
            height=100
        )
        
        col_send, col_clear = st.columns([1, 1])
        
        with col_send:
            if st.button("Send Message", type="primary", use_container_width=True):
                if user_input.strip():
                    with st.spinner(f"{agent.name} is thinking..."):
                        prompt = workspace.get_agent_prompt(selected_agent, user_input)
                        response = workspace.hf_api.query_model(model_name, prompt, max_length=200)
                        
                        # Clean up response (remove the original prompt if it's included)
                        if "Response:" in response:
                            response = response.split("Response:")[-1].strip()
                        
                        SessionMemory.add_conversation(selected_agent, user_input, response)
                        
                        st.success("Response received!")
                        st.rerun()
        
        with col_clear:
            if st.button("Clear Chat", type="secondary", use_container_width=True):
                # Clear conversations for this agent
                if 'session_memory' in st.session_state:
                    st.session_state.session_memory['conversations'] = [
                        c for c in st.session_state.session_memory['conversations'] 
                        if c['agent'] != selected_agent
                    ]
                st.rerun()
        
        # Display conversation history
        st.subheader("💬 Conversation History")
        
        history = SessionMemory.get_conversation_history(selected_agent, limit=5)
        
        if history:
            for conv in reversed(history):  # Show most recent first
                with st.expander(f"🕐 {conv['timestamp'][:19].replace('T', ' ')}"):
                    st.markdown(f"**You:** {conv['user_input']}")
                    st.markdown(f"**{workspace.agents[conv['agent']].name}:** {conv['response']}")
        else:
            st.info(f"No conversation history with {agent.name} yet. Start chatting!")
    
    with col2:
        st.header("👥 Team Overview")
        
        # Display all agents status
        for key, agent in workspace.agents.items():
            recent_activity = len([c for c in SessionMemory.get_conversation_history(key, limit=1)])
            status_class = "status-active" if recent_activity > 0 else "status-idle"
            status_text = "Active" if recent_activity > 0 else "Idle"
            
            st.markdown(f"""
            <div style="background: white; padding: 1rem; border-radius: 5px; margin: 0.5rem 0; border: 1px solid #dee2e6;">
                <h5>{agent.name}</h5>
                <p style="margin: 0.5rem 0; font-size: 0.9rem;">{agent.description}</p>
                <span class="{status_class}">● {status_text}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Quick stats
        st.header("📊 Session Stats")
        
        total_conversations = len(SessionMemory.get_conversation_history())
        active_agents = len([key for key in workspace.agents.keys() 
                           if SessionMemory.get_conversation_history(key)])
        
        st.metric("Total Messages", total_conversations)
        st.metric("Active Agents", active_agents)
        
        # Recent activity
        if total_conversations > 0:
            st.subheader("🔄 Recent Activity")
            recent = SessionMemory.get_conversation_history(limit=3)
            for conv in recent:
                agent_name = workspace.agents[conv['agent']].name
                st.markdown(f"**{agent_name}**: {conv['user_input'][:50]}...")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6c757d; font-size: 0.9rem;">
        <p>🚀 Built with Streamlit • Powered by Hugging Face • Ready for Railway deployment</p>
        <p>Mobile-optimized • Session-based memory • Modular architecture</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
