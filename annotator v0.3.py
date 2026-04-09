import fitz # PyMuPDF
import json
import ast
import time
import re
from huggingface_hub import InferenceClient
# Setup Hugging Face Client using your token
client = InferenceClient(api_key="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")
# Map categories directly to 3 specific colors
THEME_COLORS = {
    "bildungsroman_quotes": (1, 1, 0), # Yellow
    "expectations_quotes": (0.6, 1, 0.6), # Green
    "barriers_quotes": (0.6, 0.8, 1) # Blue
}
def extract_and_parse_json(text):
    """Safely extracts JSON even if the AI messes up quotes or commas."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found in response.")
   
    clean_text = match.group(0)
   
    # Try strict JSON first
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        # If the AI messed up inner quotes or commas, `ast` is much more forgiving
        try:
            # Replace true newlines with escaped newlines so string literals don't break
            clean_text = clean_text.replace('\n', '\\n')
            return ast.literal_eval(clean_text)
        except Exception as e:
            raise ValueError(f"Failed to parse JSON/Dict: {e}")
def annotate_pdf(input_pdf, output_pdf):
    doc = fitz.open(input_pdf)
   
    for page_num in range(len(doc)):
        print(f"\nProcessing page {page_num + 1} of {len(doc)}...")
        page = doc[page_num]
        text = page.get_text()
       
        if len(text.strip()) < 200:
            print("Skipping page (not enough text).")
            continue
           
        prompt = f"""
        You are an experienced reader annotating a page from "The Catcher in the Rye".
        You color-code your highlights based on your teacher's 3 rubric requirements.
       
        Task 1 (Highlights): Extract a total of 5 to 7 full, important sentences from the text below.
        Categorize them into the JSON arrays provided.
        CRITICAL: The quotes must be EXACT, word-for-word substrings from the text.
        CRITICAL: Do NOT use literal double quotes inside your strings. If the text has quotes, change them to single quotes (').
       
        Task 2 (Comments): Write exactly 2 comments about this page.
        CRITICAL RULES FOR COMMENTS:
        - They must be short, concise, academic, and incomplete sentences.
        - Max 10 to 15 words per comment
        - Make each comment unique to this page's specific events, dialogues, or thoughts, avoiding repetition across pages.
       
        Respond ONLY with a valid JSON object in this exact format:
        {{
            "bildungsroman_quotes": ["exact sentence 1 here"],
            "expectations_quotes": ["exact sentence 2 here", "exact sentence 3 here"],
            "barriers_quotes": ["exact sentence 4 here", "exact sentence 5 here", "exact sentence 6 here"],
            "comment_top": "short choppy note here",
            "comment_bottom": "another short choppy note"
        }}
       
        Page Text:
        {text}
        """
       
        max_retries = 3
        success = False
       
        # --- AUTO-RETRY LOOP ---
        for attempt in range(max_retries):
            try:
                response = client.chat_completion(
                    model="meta-llama/Meta-Llama-3-8B-Instruct",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=800
                )
               
                raw_text = response.choices[0].message.content
                data = extract_and_parse_json(raw_text)
               
                # 1. Apply Categorized Highlights
                for category, color in THEME_COLORS.items():
                    quotes = data.get(category, [])
                    if isinstance(quotes, str):
                        quotes = [quotes]
                       
                    for quote in quotes:
                        if not quote or len(quote) < 5:
                            continue
                           
                        clean_quote = quote.replace('\n', ' ').strip()
                        text_instances = page.search_for(clean_quote)
                       
                        for inst in text_instances:
                            annot = page.add_highlight_annot(inst)
                            annot.set_colors(stroke=color)
                            annot.update()
               
                # 2. Apply Top Comment
                top_comment = data.get('comment_top', '')
                if top_comment:
                    rect_top = fitz.Rect(40, 20, 300, 60)
                    annot_top = page.add_freetext_annot(
                        rect_top, top_comment, fontsize=11,
                        text_color=(0, 0, 0), fill_color=(1, 1, 0.8)
                    )
                    annot_top.update()
                   
                # 3. Apply Bottom Comment
                bottom_comment = data.get('comment_bottom', '')
                if bottom_comment:
                    page_height = page.rect.height
                    rect_bottom = fitz.Rect(300, page_height - 60, 560, page_height - 20)
                    annot_bottom = page.add_freetext_annot(
                        rect_bottom, bottom_comment, fontsize=11,
                        text_color=(0, 0, 0), fill_color=(0.8, 1, 1)
                    )
                    annot_bottom.update()
               
                print(f"✅ Success! Color-coded 3 themes with realistic notes.")
                time.sleep(3)
                success = True
                break # Break out of the retry loop if successful
               
            except Exception as e:
                print(f"⚠️ JSON Glitch on attempt {attempt + 1}. Retrying...")
                time.sleep(2) # Brief pause before retry
       
        # If it failed 3 times in a row, skip it so the program doesn't freeze
        if not success:
            print(f"❌ Failed all 3 attempts on page {page_num + 1}. Skipping to next page.")
           
    doc.save(output_pdf)
    print(f"\nDone! Saved beautifully annotated file as {output_pdf}")
# --- RUN THE SCRIPT ---
annotate_pdf("book1.pdf", "full_annotation.pdf")
