"""ACE-style roles: Generator, Reflector, and Curator prompt builders."""

import re
from typing import List, Optional, Tuple

from slm_ace.playbook import Playbook


def _get_domain_specific_instructions(domain: str) -> Optional[str]:
    """
    Get domain-specific instructions for the generator prompt.
    
    Args:
        domain: Domain name (e.g., "finance", "medical", "iot").
        
    Returns:
        Domain-specific instruction string, or None if no specific instructions.
    """
    domain_lower = domain.lower()
    
    if "finance" in domain_lower or "financial" in domain_lower:
        return "For numerical questions, show your calculations. For financial terms, provide precise definitions when relevant."
    elif "medical" in domain_lower or "health" in domain_lower:
        return "For medical questions, reference specific symptoms, conditions, or procedures mentioned in the context. Prioritize accuracy over generality."
    elif "iot" in domain_lower or "internet of things" in domain_lower:
        return "For IoT questions, consider device connectivity, protocols, and real-world constraints. Reference specific technologies mentioned in context."
    elif "technology" in domain_lower or "tech" in domain_lower:
        return "For technology questions, reference specific tools, frameworks, or concepts mentioned. Provide concrete examples when relevant."
    
    return None


def _get_domain_reflection_guidelines(domain: str) -> Optional[str]:
    """
    Get domain-specific reflection guidelines for the reflector prompt.
    
    Args:
        domain: Domain name (e.g., "finance", "medical", "iot").
        
    Returns:
        Domain-specific guideline string, or None if no specific guidelines.
    """
    domain_lower = domain.lower()
    
    if "finance" in domain_lower or "financial" in domain_lower:
        return "Focus on mathematical errors, formula misapplications, or misinterpretation of financial concepts. Reference specific calculation steps or financial principles."
    elif "medical" in domain_lower or "health" in domain_lower:
        return "Focus on diagnostic reasoning errors, symptom misinterpretations, or medical concept misunderstandings. Reference specific medical knowledge or terminology."
    elif "iot" in domain_lower or "internet of things" in domain_lower:
        return "Focus on connectivity issues, protocol misunderstandings, or device constraint oversights. Reference specific IoT architectures or technologies."
    elif "technology" in domain_lower or "tech" in domain_lower:
        return "Focus on technical concept errors, tool misapplications, or implementation oversights. Reference specific technologies or methodologies."
    
    return None


def build_generator_prompt(
    domain: str,
    playbook: Playbook,
    question: str,
    context: Optional[str] = None,
    ace_mode: str = "ace_full",
    token_budget: int = 500,
    current_step: int = 0,
) -> str:
    """
    Build a prompt for the Generator role (model that answers questions).
    
    This prompt includes:
    - Domain context
    - Top-k playbook strategies for the domain (or budget-limited for working memory)
    - The question and optional context
    
    Args:
        domain: Domain name (e.g., "finance", "medical").
        playbook: The ACE playbook.
        question: The question to answer.
        context: Optional context/background information.
        ace_mode: ACE mode ("ace_full" or "ace_working_memory").
        token_budget: Token budget for working memory mode (default: 500).
        current_step: Current step counter for recency calculation.
        
    Returns:
        Formatted prompt string.
    """
    # Get strategies from playbook based on mode
    if ace_mode == "ace_working_memory":
        # Use token-budgeted selection for working memory mode
        from slm_ace.config import ACE_MODE_WORKING
        top_strategies = playbook.get_top_entries_for_budget(
            domain=domain,
            token_budget=token_budget,
            current_step=current_step,
        )
    else:
        # Use top-k for full ACE mode
        top_strategies = playbook.get_top_k(domain, k=5, current_step=current_step)
    
    # Build domain header with domain-specific instructions
    domain_instructions = _get_domain_specific_instructions(domain)
    domain_header = f"You are an expert assistant specializing in {domain} domain questions."
    
    # Build playbook section with structured formatting
    if top_strategies:
        playbook_section = "\n\nRelevant strategies from previous experience:\n"
        for i, strategy in enumerate(top_strategies, 1):
            playbook_section += f"{i}. {strategy.text}\n"
        playbook_section += "\nUse these strategies as guidance when formulating your answer.\n"
    else:
        playbook_section = "\n\n(No prior strategies available yet. Use your domain expertise to answer.)\n"
    
    # Build question section with structured formatting
    question_section = f"\n\nQuestion:\n{question}\n"
    
    if context:
        question_section = f"\n\nContext:\n{context}\n{question_section}"
    
    # Build reasoning instructions - ALWAYS require step-by-step reasoning
    reasoning_instructions = "\n\nInstructions:\n"
    reasoning_instructions += "- Apply the relevant strategies from above when answering.\n"
    reasoning_instructions += "- You MUST show your step-by-step reasoning process before providing the final answer.\n"
    reasoning_instructions += "- Break down the problem into clear steps, showing calculations, logic, or analysis.\n"
    if domain_instructions:
        reasoning_instructions += f"- {domain_instructions}\n"
    reasoning_instructions += "\nResponse Format:\n"
    reasoning_instructions += "Reasoning:\n"
    reasoning_instructions += "[Show your step-by-step reasoning here]\n\n"
    reasoning_instructions += "Answer:\n"
    reasoning_instructions += "[Provide your final answer here]\n"
    
    prompt = f"{domain_header}{playbook_section}{question_section}{reasoning_instructions}"
    
    return prompt


def build_reflector_prompt(
    domain: str,
    question: str,
    context: Optional[str],
    model_answer: str,
    ground_truth: str,
    reasoning: Optional[str] = None,
) -> str:
    """
    Build a prompt for the Reflector role (model that generates lessons).
    
    The Reflector analyzes the model's answer against ground truth and produces
    specific, actionable lessons.
    
    Args:
        domain: Domain name.
        question: The original question.
        context: Optional context that was provided.
        model_answer: The answer generated by the model.
        ground_truth: The correct answer.
        reasoning: Optional reasoning from the model.
        
    Returns:
        Formatted prompt string.
    """
    correct = model_answer.strip().lower() == ground_truth.strip().lower()
    status = "correct" if correct else "incorrect"
    
    prompt = f"""You are analyzing a {domain} domain question-answer pair.

Question: {question}
"""
    
    if context:
        prompt += f"Context: {context}\n"
    
    prompt += f"""
Model Answer: {model_answer}
Correct Answer: {ground_truth}
Status: {status}
"""
    
    if reasoning:
        prompt += f"Model Reasoning: {reasoning}\n"
    
    domain_guidelines = _get_domain_reflection_guidelines(domain)
    
    prompt += """
Your task: Extract very specific, actionable rules that can be directly applied to future questions.

Generate 1-3 short, specific bullet-point lessons that:
- Contain concrete formulas, procedures, or specific domain knowledge (NOT vague advice)
- Include specific steps, equations, or domain-specific facts that can be directly applied
- Highlight what went wrong (if incorrect) or what specific strategy/rule worked (if correct)
- Are precise enough that someone could follow them step-by-step
"""
    
    if domain_guidelines:
        prompt += f"- {domain_guidelines}\n"
    
    prompt += """
Format your response as bullet points, one per line, starting with "-" or "•".

Examples of EXCELLENT specific rules (formulas, procedures):
- "For finance percentage calculations: convert percentage to decimal (divide by 100), then multiply: result = (percentage / 100) * base_amount"
- "For medical symptom questions: identify the specific condition mentioned, then match symptoms from context to that condition's known presentation"
- "When calculating compound interest: use formula A = P(1 + r/n)^(nt) where P=principal, r=rate, n=compounds/year, t=years"

Examples of GOOD specific rules:
- "For finance questions involving percentage calculations, always convert percentages to decimals before multiplying."
- "When medical questions ask about symptoms, cross-reference with the specific condition mentioned in the context."

Examples of BAD lessons (too vague/generic):
- "Think carefully about the question." ❌
- "Pay attention to details." ❌
- "Consider all options." ❌
- "Be thorough in your analysis." ❌
- "Use the context provided." ❌ (too vague - doesn't specify HOW)

Your lessons must contain:
- Specific formulas, equations, or procedures
- Concrete steps that can be followed
- Domain-specific knowledge or facts
- NOT generic advice or vague instructions

Avoid generic phrases entirely. Every lesson must be a concrete, actionable rule.
"""
    
    return prompt


def parse_generator_output(text: str) -> Tuple[str, Optional[str]]:
    """
    Parse the Generator's output to extract reasoning and answer.
    
    The Generator output should have the format:
    Reasoning:
    [reasoning text]
    
    Answer:
    [answer text]
    
    Args:
        text: Raw output from the Generator model.
        
    Returns:
        Tuple of (answer, reasoning). Reasoning may be None if not found.
    """
    reasoning = None
    answer = None
    
    # Try to extract reasoning and answer sections
    text_lower = text.lower()
    reasoning_keywords = ["reasoning:", "reasoning", "step-by-step:", "steps:"]
    answer_keywords = ["answer:", "answer", "final answer:", "result:"]
    
    lines = text.split("\n")
    current_section = None
    reasoning_lines = []
    answer_lines = []
    
    for line in lines:
        line_lower = line.lower().strip()
        
        # Check if this line starts a new section
        if any(keyword in line_lower for keyword in reasoning_keywords):
            current_section = "reasoning"
            # Extract text after the keyword
            for keyword in reasoning_keywords:
                if keyword in line_lower:
                    after_keyword = line[line_lower.find(keyword) + len(keyword):].strip()
                    if after_keyword:
                        reasoning_lines.append(after_keyword)
                    break
            continue
        elif any(keyword in line_lower for keyword in answer_keywords):
            current_section = "answer"
            # Extract text after the keyword
            for keyword in answer_keywords:
                if keyword in line_lower:
                    after_keyword = line[line_lower.find(keyword) + len(keyword):].strip()
                    if after_keyword:
                        answer_lines.append(after_keyword)
                    break
            continue
        
        # Add line to current section
        if current_section == "reasoning":
            if line.strip():
                reasoning_lines.append(line.strip())
        elif current_section == "answer":
            if line.strip():
                answer_lines.append(line.strip())
    
    # Build reasoning and answer strings
    if reasoning_lines:
        reasoning = "\n".join(reasoning_lines).strip()
    
    if answer_lines:
        answer = "\n".join(answer_lines).strip()
    else:
        # If no explicit answer section, try to find the last substantial line
        # or just use the whole text if no structure found
        if not reasoning and not answer:
            # No structured format found, return whole text as answer
            answer = text.strip()
        elif reasoning:
            # Has reasoning but no explicit answer section
            # Answer might be after reasoning or we need to extract it differently
            # For now, try to find the last line or last few lines as answer
            non_reasoning_lines = [l.strip() for l in lines if l.strip() and l.strip().lower() not in [k.lower() for k in reasoning_keywords + answer_keywords]]
            if non_reasoning_lines:
                # Take the last substantial line(s) as answer
                answer = non_reasoning_lines[-1]
                if len(non_reasoning_lines) > 1 and len(answer) < 20:
                    # If last line is too short, take last 2-3 lines
                    answer = "\n".join(non_reasoning_lines[-min(3, len(non_reasoning_lines)):]).strip()
    
    return answer, reasoning


def parse_reflector_output_to_lessons(text: str) -> List[str]:
    """
    Parse the Reflector's output into a list of lesson strings.
    
    Args:
        text: Raw output from the Reflector model.
        
    Returns:
        List of lesson strings (cleaned and filtered).
    """
    lessons = []
    lines = text.strip().split("\n")
    
    for line in lines:
        line = line.strip()
        # Look for bullet points
        if line.startswith("-") or line.startswith("•"):
            # Remove bullet marker and clean
            lesson = line.lstrip("-•").strip()
            if lesson:
                lessons.append(lesson)
        elif line and len(line) > 10:  # Also accept non-bullet lines if substantial
            lessons.append(line)
    
    return lessons


def build_curator_prompt(
    domain: str,
    lessons: List[str],
) -> str:
    """
    Build a prompt for the Curator role (model that marks generic rules).
    
    The Curator evaluates lessons and marks obviously generic rules as is_generic=True
    so the scoring can down-weight them.
    
    Args:
        domain: Domain name.
        lessons: List of lesson strings to evaluate.
        
    Returns:
        Formatted prompt string for curator evaluation.
    """
    lessons_text = ""
    for i, lesson in enumerate(lessons, 1):
        lessons_text += f"{i}. {lesson}\n"
    
    prompt = f"""You are a Curator evaluating lessons extracted from a {domain} domain question-answer pair.

Your task: Mark lessons that are obviously generic or vague as is_generic=True. Only specific, actionable rules with concrete procedures, formulas, or concrete steps should be marked as is_generic=False.

Lessons to evaluate:
{lessons_text}

For each lesson, determine if it is:
- SPECIFIC: Contains concrete formulas, equations, procedures, steps, or domain-specific facts
- GENERIC: Contains vague advice, generic instructions, or lacks concrete actionable details

Examples of GENERIC lessons (should be marked is_generic=True):
- "Think carefully about the question."
- "Pay attention to details."
- "Consider all options."
- "Be thorough in your analysis."
- "Use the context provided." (without specifying HOW)
- "Check your work." (without specifying WHAT to check)

Examples of SPECIFIC lessons (should be marked is_generic=False):
- "For finance percentage calculations: convert percentage to decimal (divide by 100), then multiply"
- "When calculating compound interest: use formula A = P(1 + r/n)^(nt) where P=principal, r=rate, n=compounds/year, t=years"
- "For medical symptom questions: identify the specific condition mentioned, then match symptoms from context to that condition's known presentation"

For each lesson, respond with:
Lesson [number]: is_generic=[True/False]

Be strict: if a lesson doesn't contain specific formulas, procedures, or concrete steps, mark it as generic.
"""
    
    return prompt


def parse_curator_output(lesson_count: int, text: str) -> List[bool]:
    """
    Parse the Curator's output to extract is_generic flags for each lesson.
    
    Args:
        lesson_count: Number of lessons being evaluated.
        text: Raw output from the Curator model.
        
    Returns:
        List of boolean values indicating is_generic status for each lesson (True = generic).
        Defaults to False (not generic) if parsing fails for a lesson.
    """
    is_generic_flags = [False] * lesson_count  # Default to not generic
    lines = text.strip().split("\n")
    
    for line in lines:
        line = line.strip()
        # Look for pattern: "Lesson 1: is_generic=True" or "Lesson 1: is_generic=False"
        match = re.search(r'Lesson\s+(\d+)\s*:\s*is_generic\s*=\s*(True|False)', line, re.IGNORECASE)
        if match:
            lesson_num = int(match.group(1))
            is_generic = match.group(2).lower() == "true"
            if 1 <= lesson_num <= lesson_count:
                is_generic_flags[lesson_num - 1] = is_generic  # Convert to 0-based index
    
    return is_generic_flags


def choose_lessons_for_playbook(
    domain: str,
    lessons: List[str],
    existing_playbook: Playbook,
    min_length: int = 15,
) -> List[str]:
    """
    Filter and deduplicate lessons before adding to playbook.
    
    Args:
        domain: Domain name.
        lessons: List of candidate lesson strings.
        existing_playbook: Current playbook to check against.
        min_length: Minimum character length for a lesson to be kept.
        
    Returns:
        Filtered list of lessons suitable for playbook.
    """
    filtered = []
    
    # Generic phrases to filter out
    generic_phrases = [
        "think carefully",
        "be careful",
        "pay attention",
        "consider",
        "remember",
        "make sure",
    ]
    
    for lesson in lessons:
        lesson = lesson.strip()
        
        # Filter too short
        if len(lesson) < min_length:
            continue
        
        # Filter generic
        lesson_lower = lesson.lower()
        if any(phrase in lesson_lower for phrase in generic_phrases):
            # Only skip if it's mostly generic
            if len(lesson.split()) < 5:
                continue
        
        # Check against existing entries (simple deduplication)
        is_duplicate = False
        for entry in existing_playbook.entries:
            if entry.domain == domain:
                if lesson.lower() in entry.text.lower() or entry.text.lower() in lesson.lower():
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            filtered.append(lesson)
    
    return filtered


def build_self_refine_critique_prompt(
    domain: str,
    question: str,
    context: Optional[str],
    initial_answer: str,
    ground_truth: str,
    initial_reasoning: Optional[str] = None,
) -> str:
    """
    Build a prompt for the model to critique its own answer (for self_refine mode).
    
    Args:
        domain: Domain name.
        question: The original question.
        context: Optional context that was provided.
        initial_answer: The initial answer generated by the model.
        ground_truth: The correct answer.
        initial_reasoning: Optional reasoning from the initial answer.
        
    Returns:
        Formatted prompt string for critique.
    """
    domain_instructions = _get_domain_specific_instructions(domain)
    
    prompt = f"""You are analyzing a {domain} domain question-answer pair where your initial answer was incorrect.

Question: {question}
"""
    
    if context:
        prompt += f"Context: {context}\n"
    
    prompt += f"""
Your Initial Answer: {initial_answer}
"""
    
    if initial_reasoning:
        prompt += f"Your Initial Reasoning:\n{initial_reasoning}\n"
    
    prompt += f"""
Correct Answer: {ground_truth}

Your task: Critically analyze where your reasoning or approach went wrong.

Provide a brief, specific critique that:
- Identifies the specific step, calculation, or logical error in your reasoning
- Explains what should have been done differently
- Focuses on concrete issues (wrong formula, missing step, misinterpretation), NOT generic advice
"""
    
    if domain_instructions:
        prompt += f"\n{domain_instructions}\n"
    
    prompt += """
Critique:
"""
    
    return prompt


def build_self_refine_rewrite_prompt(
    domain: str,
    question: str,
    context: Optional[str],
    initial_answer: str,
    critique: str,
    ground_truth: str,
) -> str:
    """
    Build a prompt for the model to rewrite its answer after seeing critique (for self_refine mode).
    
    Args:
        domain: Domain name.
        question: The original question.
        context: Optional context that was provided.
        initial_answer: The initial answer generated by the model.
        critique: The critique of the initial answer.
        ground_truth: The correct answer.
        
    Returns:
        Formatted prompt string for rewriting.
    """
    domain_instructions = _get_domain_specific_instructions(domain)
    
    prompt = f"""You are answering a {domain} domain question. You previously gave an incorrect answer, received critique, and now see the correct answer.

Question: {question}
"""
    
    if context:
        prompt += f"Context: {context}\n"
    
    prompt += f"""
Your Initial Answer: {initial_answer}
Critique: {critique}
Correct Answer: {ground_truth}

Your task: Provide a corrected answer with clear step-by-step reasoning.

You MUST:
1. Show your step-by-step reasoning process before providing the final answer
2. Break down the problem into clear steps, showing calculations, logic, or analysis
3. Address the specific issues mentioned in the critique
4. Apply the correct approach based on the critique and ground truth
"""
    
    if domain_instructions:
        prompt += f"\n{domain_instructions}\n"
    
    prompt += """
Response Format:
Reasoning:
[Show your step-by-step reasoning here, addressing the critique]

Answer:
[Provide your final, corrected answer here]
"""
    
    return prompt

