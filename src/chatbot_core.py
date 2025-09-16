import torch
import transformers
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    pipeline
)
from peft import PeftModel
from langchain.memory import ConversationSummaryMemory
from langchain_huggingface import HuggingFacePipeline
import re
import json

class ClinicalChatbot:
    def __init__(self, model_path: str = "Final_Model"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.llm = None
        self.sessions = {}  # Store conversation sessions
        
    def load_model_and_tokenizer(self):
        """Load the fine-tuned model and tokenizer"""
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        model = AutoModelForCausalLM.from_pretrained(
            "BioMistral/BioMistral-7B",
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        model = PeftModel.from_pretrained(model, self.model_path)
        
        tokenizer = AutoTokenizer.from_pretrained(
            "BioMistral/BioMistral-7B",
            trust_remote_code=True
        )
        
        # Create pipeline for langchain LLM
        pipeline_obj = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=2048,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        llm = HuggingFacePipeline(pipeline=pipeline_obj)
        
        return model, tokenizer, pipeline_obj, llm
    
    def initialize_session(self, session_id: str):
        """Initialize a new conversation session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'memory': ConversationSummaryMemory(llm=self.llm),
                'last_activity': datetime.now()
            }
        return self.sessions[session_id]
    
    def cleanup_old_sessions(self, max_age_minutes: int = 60):
        """Clean up old sessions to prevent memory leaks"""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session_data in self.sessions.items():
            session_age = (current_time - session_data['last_activity']).total_seconds() / 60
            if session_age > max_age_minutes:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
    
    def format_input(self, input_text: str, memory):
        """Format the input with system prompt and chat history"""
        system_prompt = """You are an AI clinical decision support tool for a neurocritical care team managing severe TBI patients in real-time. Your role is to analyze provided clinical data and suggest immediate, evidence-based actions.

**CRITICAL INSTRUCTIONS:**
1.  **Base Analysis Solely on Provided Data:** Only use the clinical features provided in the immediate input. Do not infer or assume missing data. If data is insufficient for a high-confidence recommendation, state so and recommend necessary monitoring to obtain that data.
2.  **Adhere to Protocols:** Strictly base recommendations on the BOOST-3 and BONANZA protocols. Do not reference other protocols unless they are explicitly mentioned in the input.
3.  **Active Voice & Clinical Tone:** Phrase all recommendations in the active, imperative mood as direct instructions to the clinical team (e.g., "Administer...", "Increase...", "Monitor..."). Avoid past tense and speculative language.
4.  **Be Concise and Specific:** Avoid repetition, fluff, and irrelevant details. Recommendations must be specific and actionable (include drug, dose, route where applicable).
5.  **NO HALLUCINATION:** Do not generate or invent monitoring parameters, lab values, or clinical findings that are not present in the input. If it's not provided, do not include it.
6.  **Output Format:** Respond **ONLY** with a valid JSON object. Do not use markdown, code blocks, or any other formatting. Do not add any text before or after the JSON.

**Confidence Level Guidance:**
- **HIGH (>80%):** Provided data is classic and unambiguous for a specific protocol-mandated intervention.
- **MODERATE (50-80%):** Data suggests an intervention, but some ambiguity or missing data exists.
- **LOW (<50%):** Data is insufficient or conflicting; recommendation is for cautious monitoring or gathering more data.

**Output a JSON object with the following exact structure:**

{
  "confidence": "[CONFIDENCE_LEVEL] [X]% confidence",
  "recommended_actions": ["Action 1: Specific instruction with dosage if applicable.", "Action 2: Another specific instruction."],
  "monitoring": ["Parameter 1: Frequency (e.g., ICP: every 5 min)", "Parameter 2: Frequency"],
  "recovery_probability": "[X]%",
  "clinical_reasoning": "Concise, active-voice rationale. First, state the acute problem from the data. Second, cite the specific protocol recommendation that applies. Third, link the proposed intervention to the expected pathophysiological effect. Keep this section under 3 sentences."
}
"""
        
        # Getting conversation history
        history = memory.load_memory_variables({})["history"]
        
        # Building the full prompt
        if history:
            full_prompt = f"{system_prompt}\n\nConversation History:\n{history}\n\nCurrent Input:\n{input_text}"
        else:
            full_prompt = f"{system_prompt}\n\nInput:\n{input_text}"
        
        return [{"role": "user", "content": full_prompt}]
    
    def generate_response(self, formatted_input):
        """Generate model response"""
        prompt = self.tokenizer.apply_chat_template(
            formatted_input,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extracting only the assistant's response
        if "[/INST]" in response:
            return response.split("[/INST]")[-1].strip()
        else:
            return response
    
    def clean_text(self, text):
        """Clean text by replacing problematic characters"""
        # Firstly decoding any Unicode escape sequences
        text = text.encode().decode('unicode_escape')
        
        replacements = {
            "â€”": "—",  # em dash
            "â†’": "→",  # right arrow
            "â†“": "↓",  # down arrow
            "â†‘": "↑",  # up arrow
            "â‚‚": "₂",  # subscript 2
            "â€“": "–",  # en dash
            "\\u2014": "—",  # em dash
            "\\u2082": "₂",  # subscript 2
            "\\u00b0": "°",  # degree symbol
        }
        
        for wrong, right in replacements.items():
            text = text.replace(wrong, right)
        
        return text
    
    def process_message(self, message: str, session_id: str = "default"):
        """Process a message through the chatbot"""
        # Initialize or get session
        session = self.initialize_session(session_id)
        memory = session['memory']
        session['last_activity'] = datetime.now()
        
        # Clean input text
        cleaned_input = self.clean_text(message)
        
        # Format input with conversation history
        formatted_input = self.format_input(cleaned_input, memory)
        
        # Generate response
        predicted_output = self.generate_response(formatted_input)
        
        # Clean predicted output
        cleaned_output = self.clean_text(predicted_output)
        
        # Save conversation to memory
        memory.save_context({"input": message}, {"output": cleaned_output})
        
        return cleaned_output