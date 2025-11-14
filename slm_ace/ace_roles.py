"""ACE-style roles: Generator, Reflector, and Curator prompt builders."""

import re
from typing import List, Optional, Tuple

from slm_ace.playbook import Playbook


def detect_question_type(question: str, context: Optional[str] = None) -> str:
    """
    Detect the type of question (calculation, extraction, reasoning, general).
    
    Args:
        question: The question text.
        context: Optional context.
        
    Returns:
        Task type string.
    """
    combined = f"{question} {context or ''}".lower()
    
    # Finance-specific patterns
    calc_keywords = [
        "calculate", "compute", "total", "sum", "difference", "profit", "margin",
        "percentage", "rate", "subtract", "add", "multiply", "divide", "after tax",
        "before tax", "net", "gross", "revenue", "expense", "cost"
    ]
    extract_keywords = [
        "what is", "find", "identify", "extract", "locate", "search", "what was",
        "which", "where", "when"
    ]
    reason_keywords = [
        "why", "how", "explain", "reason", "because", "due to", "caused by"
    ]
    
    if any(keyword in combined for keyword in calc_keywords):
        return "calculation"
    elif any(keyword in combined for keyword in extract_keywords):
        return "extraction"
    elif any(keyword in combined for keyword in reason_keywords):
        return "reasoning"
    else:
        return "general"


def build_generator_prompt(
    domain: str,
    playbook: Playbook,
    question: str,
    context: Optional[str] = None,
) -> str:
    """
    Build a task-aware prompt for the Generator role, optimized for finance/TAT-QA.
    
    This prompt includes:
    - Domain-specific context and instructions
    - Task-aware playbook strategies (selected by question type)
    - Structured formatting for better comprehension
    - Domain-specific guidance (especially for finance calculations)
    
    Args:
        domain: Domain name (e.g., "finance", "medical").
        playbook: The ACE playbook.
        question: The question to answer.
        context: Optional context/background information.
        
    Returns:
        Formatted prompt string optimized for the task.
    """
    # Detect question type for task-aware strategy selection
    task_type = detect_question_type(question, context)
    
    # Get task-aware strategies from playbook
    top_strategies = playbook.get_top_k(
        domain=domain,
        k=5,
        question=question,
        context=context,
        task_type=task_type,
    )
    
    # Build domain-specific header with task awareness
    if domain == "finance":
        domain_header = """You are an expert financial analyst specializing in TAT-QA (Table and Text Question Answering) tasks.

Your task is to:
1. Carefully read and understand the financial context provided
2. Identify relevant numerical values (revenue, expenses, tax rates, etc.)
3. Perform accurate calculations when needed
4. Extract specific information when asked
5. Provide precise numerical answers without units or formatting (e.g., "100000" not "$100,000")

Key principles:
- For calculation questions: Break down the problem step-by-step (e.g., revenue - expenses = profit, then profit - tax = net profit)
- For extraction questions: Locate the exact value in the context
- Always verify your answer matches the question's requirements
- Pay attention to "before tax" vs "after tax" distinctions
- For percentages: Calculate as (value / total) * 100"""
    else:
        domain_header = f"You are an expert assistant specializing in {domain} domain questions."
    
    # Build playbook section with better formatting
    if top_strategies:
        playbook_section = "\n\n=== Proven Strategies from Experience ===\n"
        for i, strategy in enumerate(top_strategies, 1):
            strategy_type = getattr(strategy, 'task_type', 'general')
            playbook_section += f"{i}. [{strategy_type.upper()}] {strategy.text}\n"
        playbook_section += "\nApply these strategies when relevant to the current question.\n"
    else:
        playbook_section = "\n\n(No prior strategies available yet. You'll learn from this experience.)\n"
    
    # Build context section with structured formatting
    context_section = ""
    if context:
        context_section = f"\n=== Context ===\n{context}\n"
    
    # Build question section
    question_section = f"\n=== Question ===\n{question}\n"
    
    # Add task-specific instructions
    task_instructions = ""
    if domain == "finance" and task_type == "calculation":
        task_instructions = "\n[Calculation Task] Break down the problem into steps:\n1. Identify the values needed\n2. Determine the calculation formula\n3. Perform the calculation\n4. Verify the result\n"
    elif domain == "finance" and task_type == "extraction":
        task_instructions = "\n[Extraction Task] Locate the specific value or information in the context.\n"
    
    # Build final prompt
    prompt = f"{domain_header}{playbook_section}{context_section}{question_section}{task_instructions}\n=== Answer ===\n"
    
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
    Build a task-aware prompt for the Reflector role that generates specific, actionable lessons.
    
    The Reflector analyzes the model's answer against ground truth and produces
    domain-specific, actionable lessons that can improve future performance.
    
    Args:
        domain: Domain name.
        question: The original question.
        context: Optional context that was provided.
        model_answer: The answer generated by the model.
        ground_truth: The correct answer.
        reasoning: Optional reasoning from the model.
        
    Returns:
        Formatted prompt string optimized for generating actionable lessons.
    """
    correct = model_answer.strip().lower() == ground_truth.strip().lower()
    status = "CORRECT" if correct else "INCORRECT"
    task_type = detect_question_type(question, context)
    
    # Domain-specific reflection guidance
    if domain == "finance":
        domain_guidance = """
For finance/TAT-QA tasks, focus on:
- Calculation errors: Identify which step was wrong (e.g., forgot to subtract tax, used wrong formula)
- Extraction errors: Note what information was missed or misread
- Numerical precision: Note if answer format was wrong (e.g., included "$" or commas)
- Context understanding: Note if key financial terms were misunderstood
"""
    else:
        domain_guidance = f"For {domain} tasks, focus on domain-specific patterns and common errors."
    
    # Examples of good vs bad lessons
    good_examples = """
Good lesson examples:
- "For profit margin questions, calculate as (profit / revenue) * 100, not (profit / expenses) * 100"
- "When question asks 'after tax', first calculate profit (revenue - expenses), then subtract tax (profit * tax_rate)"
- "Extract revenue values directly from context when question asks 'what is the revenue' - no calculation needed"

Bad lesson examples (too generic - avoid these):
- "Think carefully about the question"
- "Pay attention to details"
- "Be more accurate"
"""
    
    prompt = f"""You are an expert analyst reviewing a {domain} domain question-answer pair to extract actionable lessons.

=== Task Information ===
Question: {question}
"""
    
    if context:
        prompt += f"Context: {context}\n"
    
    prompt += f"""
Model Answer: {model_answer}
Correct Answer: {ground_truth}
Status: {status}
Task Type: {task_type}
"""
    
    if reasoning:
        prompt += f"Model's Reasoning: {reasoning}\n"
    
    prompt += f"""
=== Your Task ===
Analyze this example and generate 1-3 specific, actionable lessons that will help with similar questions in the future.

{domain_guidance}

{good_examples}

=== Instructions ===
Generate lessons that:
1. Are SPECIFIC and ACTIONABLE (not generic advice)
2. Include the exact pattern or formula that should be used
3. Explain what went wrong (if incorrect) or what worked well (if correct)
4. Can be directly applied to future similar questions
5. Are domain-specific and task-type specific ({task_type} tasks)

Format: One lesson per line, starting with "-" or "•"
Each lesson should be 15-80 characters and self-contained.

=== Lessons ===
"""
    
    return prompt


def parse_reflector_output_to_lessons(text: str) -> List[str]:
    """
    Parse the Reflector's output into a list of lesson strings with improved extraction.
    
    Args:
        text: Raw output from the Reflector model.
        
    Returns:
        List of lesson strings (cleaned and filtered).
    """
    lessons = []
    lines = text.strip().split("\n")
    
    # Patterns to identify lesson lines
    bullet_patterns = [
        r"^[-•]\s*(.+)$",  # Standard bullets
        r"^\d+[.)]\s*(.+)$",  # Numbered lists
        r"^[-•]\s*(.+)$",  # Alternative bullet styles
    ]
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Skip section headers
        if line.startswith("===") or line.upper() == line and len(line) > 5:
            continue
        
        # Try to extract lesson from bullet points
        lesson_extracted = None
        for pattern in bullet_patterns:
            match = re.match(pattern, line)
            if match:
                lesson_extracted = match.group(1).strip()
                break
        
        if lesson_extracted:
            lessons.append(lesson_extracted)
        elif len(line) > 15 and not line.startswith("Good") and not line.startswith("Bad"):
            # Accept substantial non-bullet lines that aren't examples
            # Filter out common prefixes
            if not any(line.startswith(prefix) for prefix in ["Example", "Note:", "Tip:", "Hint:"]):
                lessons.append(line)
    
    return lessons


def choose_lessons_for_playbook(
    domain: str,
    lessons: List[str],
    existing_playbook: Playbook,
    min_length: int = 15,
    max_length: int = 200,
) -> List[str]:
    """
    Filter and deduplicate lessons before adding to playbook with improved quality checks.
    
    Args:
        domain: Domain name.
        lessons: List of candidate lesson strings.
        existing_playbook: Current playbook to check against.
        min_length: Minimum character length for a lesson to be kept.
        max_length: Maximum character length for a lesson to be kept.
        
    Returns:
        Filtered list of lessons suitable for playbook.
    """
    filtered = []
    
    # Generic phrases to filter out (expanded list)
    generic_phrases = [
        "think carefully",
        "be careful",
        "pay attention",
        "consider",
        "remember",
        "make sure",
        "take care",
        "be aware",
        "keep in mind",
        "note that",
        "it is important",
    ]
    
    # Low-value patterns (too vague)
    vague_patterns = [
        r"^.*\b(should|must|need to|have to)\b.*$",  # Too imperative without specifics
    ]
    
    for lesson in lessons:
        lesson = lesson.strip()
        
        # Filter too short or too long
        if len(lesson) < min_length or len(lesson) > max_length:
            continue
        
        # Filter generic phrases (only if lesson is mostly generic)
        lesson_lower = lesson.lower()
        generic_word_count = sum(1 for phrase in generic_phrases if phrase in lesson_lower)
        total_words = len(lesson.split())
        
        # Skip if more than 30% of words are generic phrases
        if generic_word_count > 0 and total_words < 8:
            continue
        
        # Filter vague patterns
        is_vague = False
        for pattern in vague_patterns:
            if re.match(pattern, lesson_lower) and total_words < 6:
                is_vague = True
                break
        if is_vague:
            continue
        
        # Check for domain-specific value
        # Finance lessons should mention financial terms or calculations
        if domain == "finance":
            finance_keywords = [
                "revenue", "expense", "profit", "tax", "calculate", "formula",
                "margin", "percentage", "subtract", "add", "multiply", "divide",
                "net", "gross", "before", "after"
            ]
            if not any(keyword in lesson_lower for keyword in finance_keywords):
                # Allow if it's clearly about extraction or reasoning
                if not any(word in lesson_lower for word in ["extract", "find", "locate", "identify", "reason", "step"]):
                    continue
        
        # Check against existing entries using semantic similarity
        is_duplicate = False
        for entry in existing_playbook.entries:
            if entry.domain == domain:
                # Use playbook's semantic similarity method
                similarity = existing_playbook._compute_semantic_similarity(lesson, entry.text)
                # Also check containment for very similar entries
                if similarity > 0.7 or (
                    lesson.lower() in entry.text.lower() or entry.text.lower() in lesson.lower()
                ) and len(lesson) > 20:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            filtered.append(lesson)
    
    return filtered

