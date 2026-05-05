"""
Ollama + Groq コード修正パイプライン
=====================================
使い方:
  python ollama_groq_pipeline.py --file "ファイルパス" --design "設計内容"
  または
  python ollama_groq_pipeline.py  <- 対話モード

フロー:
1. Claudeの設計・提案をGroq(70B)に渡す（最初の1回だけ）
2. Ollamaがdiff出力（納得いくまでループ）
3. Groq(8B)がdiffをチェック → 指摘をdiffで出力
4. Groq(70B)が指摘diffと設計を比較 → OK/NG
5. NGならOllamaに差し戻し
"""

import sys
import os
import re
import json
import argparse
import requests
import difflib
import time
import ast
import subprocess

# ============================
# 設定
# ============================
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5-coder:7b-instruct-q4_K_M"

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL_CHECK = "llama-3.1-8b-instant"      # diffチェック用（8B・制限緩い）
GROQ_MODEL_FINAL = "llama-3.3-70b-versatile"   # 最終確認用（70B・高精度）
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ============================
# ユーティリティ
# ============================

def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write_file(path: str, content: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def compute_diff(before: str, after: str, filename: str = "file") -> str:
    diff = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"{filename} (修正前)",
        tofile=f"{filename} (修正後)",
    )
    return "".join(diff)

def extract_changed_lines(before: str, after: str) -> list:
    matcher = difflib.SequenceMatcher(None, before.splitlines(), after.splitlines())
    ranges = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            ranges.append((j1 + 1, j2 if j2 > j1 else j1 + 1))
    return ranges

def extract_code_block(text: str) -> str:
    # ```python ... ``` を取り出す
    match = re.search(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # <CODE> ... </CODE> も取り出せるようにする
    match = re.search(r"<CODE>\n(.*?)</CODE>", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return text.strip()

def format_line_ranges(ranges: list) -> str:
    return ", ".join(f"{s}~{e}行目" for s, e in ranges)
def make_short_diff(diff_text: str, max_lines: int = 200) -> str:
    lines = diff_text.splitlines()
    if len(lines) <= max_lines:
        return diff_text
    return "\n".join(lines[:max_lines]) + "\n\n...（diffが長いため省略）"
def extract_minimal_diff(diff_text: str, max_lines: int = 100) -> str:
    lines = diff_text.splitlines()
    filtered = []

    for line in lines:
        if line.startswith("+") or line.startswith("-"):
            filtered.append(line)

    if len(filtered) == 0:
        return diff_text[:1000]

    return "\n".join(filtered[:max_lines])

def check_python_syntax(code: str) -> tuple:
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"{e.msg} at line {e.lineno}"
    
import multiprocessing

def run_pytest(timeout: int = 30) -> tuple:
    try:
        result = subprocess.run(
            ["pytest", "--cov=.", "--cov-report=term", "--disable-warnings"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout
        )

        if result.returncode == 0:
            return True, result.stdout

        return False, result.stdout + "\n" + result.stderr

    except subprocess.TimeoutExpired:
        return False, "pytest タイムアウト"
    except FileNotFoundError:
        return False, "pytest が見つかりません。pip install pytest を実行してください。"
    
def groq_generate_pytest(code: str) -> str:
    print("\n[Groq 8B] pytestを自動生成中...")

    messages = [
        {
            "role": "system",
            "content": "あなたはPythonのpytestを書く専門家です。pytestコードだけを出力してください。説明は不要です。"
        },
        {
            "role": "user",
            "content": f"""以下のPythonコードに対するpytestテストを書いてください。
- pytest形式
- 正常系と異常系を含める
- 外部APIや危険な処理はモックまたは避ける
- コードブロックのみで返す

対象コード:
```python
{code}
```"""
        }
    ]

    response = call_groq(messages, GROQ_MODEL_CHECK)
    return extract_code_block(response)

def summarize_pytest_error(test_error: str) -> str:
    lines = test_error.splitlines()

    important = []
    for line in lines:
        if (
            "FAILED" in line
            or "ERROR" in line
            or "AssertionError" in line
            or "Traceback" in line
            or "E   " in line
        ):
            important.append(line)

    if not important:
        important = lines[-30:]

    return "\n".join(important[:80])

def parse_pytest_error_to_json(test_error: str) -> str:
    lines = test_error.splitlines()

    failed_tests = []
    current_test = None
    error_lines = []

    for line in lines:
        if line.startswith("FAILED ") or " FAILED " in line:
            if current_test:
                error_text = "\n".join(error_lines[-10:])

                failed_tests.append({
                    "test": current_test,
                    "error": error_text,
                    "expected_actual": extract_expected_actual(error_text)
                })
                error_lines = []

            current_test = line.strip()

        elif (
            "AssertionError" in line
            or "Traceback" in line
            or line.startswith("E   ")
            or "ERROR" in line
        ):
            error_lines.append(line.strip())

    if current_test:
        error_text = "\n".join(error_lines[-10:])

        failed_tests.append({
            "test": current_test,
            "error": error_text,
            "expected_actual": extract_expected_actual(error_text)
        })

    if not failed_tests:
        failed_tests.append({
            "test": "unknown",
            "error": summarize_pytest_error(test_error)
        })

    data = {
        "status": "failed",
        "failed_count": len(failed_tests),
        "failed_tests": failed_tests
    }

    return json.dumps(data, ensure_ascii=False, indent=2)

def optimize_structured_error(structured_error: str, max_tests: int = 3) -> str:
    data = json.loads(structured_error)

    # 上位N件だけ残す
    data["failed_tests"] = data["failed_tests"][:max_tests]

    # エラーも短くする
    for test in data["failed_tests"]:
        lines = test["error"].splitlines()
        test["error"] = "\n".join(lines[:5])

    return json.dumps(data, ensure_ascii=False, indent=2)

def extract_expected_actual(error_text: str) -> dict:
    patterns = [
        r"assert (.+?) == (.+)",
        r"E\s+assert (.+?) == (.+)",
        r"Expected: (.+)",
        r"Actual: (.+)",
    ]

    result = {
        "expected": None,
        "actual": None
    }

    for line in error_text.splitlines():
        m = re.search(r"assert (.+?) == (.+)", line)
        if m:
            result["actual"] = m.group(1).strip()
            result["expected"] = m.group(2).strip()

    return result

def extract_context(code: str, design: str = "", max_lines: int = 160) -> str:
    lines = code.splitlines()

    if len(lines) <= max_lines:
        return code

    # import抽出
    import_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            import_lines.append(line)

    # call graph
    try:
        call_graph = build_call_graph(code)
    except Exception:
        call_graph = {}

    # キーワード抽出
    words = [
        w for w in re.split(r"\W+", design.lower())
        if len(w) >= 3
    ]

    # seed関数
    seed_names = []

    for i, line in enumerate(lines):
        m = re.match(r"^(def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", line)
        if m:
            name = m.group(2)

            name_lower = name.lower()
            if any(w in name_lower for w in words) or any(name_lower in w for w in words):
                seed_names.append(name)

    # 依存展開
    related_names = expand_related_functions(seed_names, call_graph, depth=2)

    # 対象行
    candidates = []
    for i, line in enumerate(lines):
        m = re.match(r"^(def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", line)
        if m:
            name = m.group(2)
            if name in related_names:
                candidates.append(i)

    # コンテキスト生成
    context_blocks = []
    padding = 25

    for idx in candidates:
        start = max(0, idx - padding)
        end = min(len(lines), idx + padding)
        context_blocks.append("\n".join(lines[start:end]))

    if context_blocks:
        return (
            "\n".join(import_lines)
            + "\n\n# ===== 関連コード =====\n"
            + "\n\n# --- block ---\n\n".join(context_blocks)
        )

    return "\n".join(lines[:max_lines]) + "\n\n# ...省略..."

def extract_diff_context(code: str, changed_ranges: list, padding: int = 8) -> str:
    """
    変更行の最小コンテキストだけを抜き出す（diff完全版）
    """
    lines = code.splitlines()
    blocks = []

    for start, end in changed_ranges:
        s = max(1, start - padding)
        e = min(len(lines), end + padding)

        block = "\n".join(lines[s-1:e])
        blocks.append(block)

    return "\n\n# --- minimal diff context ---\n\n".join(blocks)

def build_call_graph(code: str) -> dict:
    """
    関数ごとの呼び出し関数一覧を作る
    """
    tree = ast.parse(code)
    graph = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            calls = set()

            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Name):
                        calls.add(child.func.id)
                    elif isinstance(child.func, ast.Attribute):
                        calls.add(child.func.attr)

            graph[node.name] = calls

    return graph


def expand_related_functions(seed_names: list, call_graph: dict, depth: int = 1) -> set:
    """
    対象関数から呼び出している関数も追加する
    """
    related = set(seed_names)

    for _ in range(depth):
        new_items = set()

        for name in related:
            for called in call_graph.get(name, []):
                if called in call_graph:
                    new_items.add(called)

        related |= new_items

    return related

def is_test_weak(test_code: str) -> bool:
    lines = test_code.splitlines()

    assert_count = sum(1 for l in lines if "assert" in l)
    has_exception = "pytest.raises" in test_code
    test_funcs = sum(1 for l in lines if l.strip().startswith("def test_"))

    if assert_count < 3:
        return True

    if test_funcs < 2:
        return True

    if not has_exception:
        return True

    return False

def explain_test_weakness(test_code: str) -> str:
    reasons = []

    if test_code.count("assert") < 3:
        reasons.append("assertが3個未満です")

    if test_code.count("def test_") < 2:
        reasons.append("テスト関数が2個未満です")

    if "pytest.raises" not in test_code:
        reasons.append("異常系テスト pytest.raises がありません")

    if not reasons:
        return "テストが弱い可能性があります"

    return "\n".join(reasons)

# ============================
# Ollama呼び出し
# ============================

def call_ollama(messages: list) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
    }

    max_retries = 5
    retry_count = 0

    while retry_count < max_retries:
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()["message"]["content"]

        except requests.exceptions.Timeout:
            print("  -> Ollamaタイムアウト、再試行...")
            retry_count += 1

        except requests.exceptions.ConnectionError:
            print("  -> Ollama接続エラー、Ollamaが起動しているか確認してください。再試行...")
            time.sleep(5)
            retry_count += 1

    raise Exception("Ollama API リトライ上限に達しました")

def extract_coverage(output: str) -> int:
    m = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", output)
    if m:
        return int(m.group(1))
    return 0

# ============================
# Groq呼び出し（レート制限対応・自動リトライ）
# ============================

def call_groq(messages: list, model: str = GROQ_MODEL_CHECK) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
    }

    max_retries = 5
    retry_count = 0

    while retry_count < max_retries:
        try:
            resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=60)
            if resp.status_code == 429:
                print("  -> レート制限、30秒待機してリトライ...")
                time.sleep(30)
                retry_count += 1
                continue

            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except requests.exceptions.Timeout:
            print("  -> タイムアウト、再試行...")
            retry_count += 1
    raise Exception("Groq API リトライ上限に達しました")

def extract_ultra_diff(diff_text: str, max_lines: int = 60) -> str:
    lines = diff_text.splitlines()

    result = []
    context_buffer = []

    for line in lines:
        if line.startswith("+") or line.startswith("-"):
            # 前の2行コンテキスト
            result.extend(context_buffer[-2:])
            result.append(line)
            context_buffer = []
        else:
            context_buffer.append(line)

    return "\n".join(result[:max_lines])

# ============================
# 設計をOllamaで分割
# ============================

def split_design(design: str) -> list:
    print("\n[Ollama] 設計を分析中...")
    messages = [
        {
            "role": "system",
            "content": "あなたはコード修正の設計を分析する専門家です。"
        },
        {
            "role": "user",
            "content": f"""以下の設計を独立した修正タスクに分割してください。
各タスクは独立して実行できる単位にしてください。
必ずJSON配列で返してください。例：["タスク1", "タスク2", "タスク3"]
JSONのみ返してください。説明は不要です。

設計：
{design}"""
        }
    ]

    response = call_ollama(messages)
    try:
        tasks = json.loads(response.strip())
        if isinstance(tasks, list) and len(tasks) > 0:
            print(f"  -> {len(tasks)}個のタスクに分割しました")
            for i, t in enumerate(tasks):
                print(f"  {i+1}. {t}")
            return tasks
    except Exception:
        pass

    print("  -> 分割不要、そのまま実行")
    return [design]

def should_run_ai(diff_text: str, min_changed_lines: int = 10) -> bool:
    lines = diff_text.splitlines()
    changed = sum(
        1 for l in lines
        if (l.startswith("+") or l.startswith("-"))
        and not l.startswith("+++")
        and not l.startswith("---")
    )
    return changed >= min_changed_lines

# ============================
# ステップ1: Ollamaが修正してdiff出力（ループ）
# ============================

def ollama_generate_diff(design: str, original_code: str, filename: str):
    print("\n[Ollama] 修正箇所を特定中...")

    system_prompt = """あなたはコード修正・追加の専門家です。
設計・提案に基づいてコードを修正または追加してください。

必ず以下のルールを守ってください：
- 修正・追加後のコード全体をコードブロック(```)で出力する
- 修正・追加した行番号を必ず明記する
- 余計な説明は禁止

追加ルール（重要）：
- 無関係なコードは絶対に変更しない
- 既存の関数名・変数名を勝手に変更しない
- テストを通すためだけの不自然な実装は禁止
- 修正は必要最小限にする
- 前回と同じ修正を繰り返さない
"""

    target_context = extract_context(original_code, design)

    messages = [
        {"role": "system", "content": system_prompt},
    {
        "role": "user",
        "content": (
            "## 設計・提案\n"
            f"{design}\n\n"
            "## 修正対象コード（抜粋）\n"
            "```python\n"
            f"{target_context}\n"
            "```\n\n"
            "必ず修正後のコード全体を出力してください\n"
            "省略せず完全なコードにしてください\n\n"
            "上記の設計に基づいてコードを修正または追加してください。"
        )
    }
]
    
    iteration = 0
    current_code = original_code
    iteration_no_change = 0

    MAX_OLLAMA_LOOP = 5
    iteration = 0

    while iteration < MAX_OLLAMA_LOOP:
        iteration += 1
        print(f"  -> Ollama修正試行 {iteration}回目...")

        response = call_ollama(messages)
        modified_code = extract_code_block(response)

        changed_ranges = extract_changed_lines(current_code, modified_code)
        diff_context = extract_diff_context(current_code, changed_ranges)
        if not changed_ranges:
            iteration_no_change += 1
            if iteration_no_change >= 5:
                print("  -> 5回変更なし、設計をより具体的にしてください")
                return None, None
            print("  -> 変更なし、再試行...")
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": """直前の修正差分で問題がある箇所だけを修正してください。
                無関係なコードは変更しないでください。
                必ず修正後のコード全体をコードブロックで出力してください。"""
            })          
            continue 

        diff_text = compute_diff(current_code, modified_code, filename)
        line_info = format_line_ranges(changed_ranges)
        print(f"  -> 変更検出: {line_info}")

        messages.append({"role": "assistant", "content": response})
        messages.append({
            "role": "user",
            "content": f"修正内容を確認してください。\n変更箇所: {line_info}\n\ndiff:\n{diff_text}\n\nこの修正は設計・提案と一致していますか？問題なければ「OK」と答えてください。問題があれば修正したコード全体を再出力してください。"
        })

        check_response = call_ollama(messages)
        print(f"  -> Ollamaチェック: {check_response[:80]}...")

        if "ok" in check_response.lower() or "問題ない" in check_response or "一致" in check_response:
            print("  -> Ollama納得！次のステップへ")
            return diff_text, modified_code

        new_code_attempt = extract_code_block(check_response)
        if new_code_attempt and new_code_attempt != check_response:
            current_code = new_code_attempt
            messages.append({"role": "assistant", "content": check_response})
            messages.append({
                "role": "user",
                "content": f"""直前の修正差分で問題がある箇所だけを修正してください。
無関係なコードは変更しないでください。

## 修正対象の周辺コード
```python
{diff_context}
```
必ず修正後のコード全体をコードブロックで出力してください。"""
})
        else:
            print("  -> Ollamaの再修正なし、現在の修正で進む")
            return diff_text, modified_code
    print("❌ 最大試行回数に到達（打ち切り）")
    return None, None   

# ============================
# ステップ2: Groq(8B)がdiffをチェック → 指摘をdiffで出力
# ============================

def groq_check_diff(diff_text: str) -> tuple:
    """
    8BがOllamaのdiffをチェックして指摘を返す。
    OK → (True, None)
    NG → (False, 指摘内容)
    """
    print("\n[Groq 8B] diffをチェック中...")

    short_diff = extract_minimal_diff(diff_text, max_lines=40)

    messages = [
        {
            "role": "system",
            "content": """あなたはコードレビューの専門家です。
diffを見て問題点を簡潔に指摘してください。

以下も問題として扱ってください:
- 無関係な変更
- 既存仕様の破壊
- テストだけを通すための不自然な実装
- 不要なimport変更
- 関数名・変数名の変更
"""        },
        {
            "role": "user",
            "content": f"## Ollamaの修正diff\n```\n{short_diff}\n```\n\nこのdiffに問題がありますか？\n- 問題なし: 「OK」とだけ答えてください\n- 問題あり: 問題点を箇条書きで簡潔に説明してください（コードは不要）"
        }
    ]

    response = call_groq(messages, GROQ_MODEL_CHECK)
    print(f"  -> 8Bチェック結果: {response[:80]}...")

    if "ok" in response.strip().lower():
        print("  -> Groq 8B: OK！")
        return True, None

    print(f"  -> Groq 8B: 問題あり、70Bに送ります")
    return False, response

# ============================
# ステップ3: Groq(70B)が指摘diffと設計を比較して最終判断
# ============================

def groq_verify_final(diff_text: str, check_result: str, modified_code: str, groq_context_final: list) -> tuple:
    """
    70Bが指摘diffと設計を比較して最終判断。
    一致 → (True, modified_code)
    ズレ → (False, None) ← Ollamaに差し戻し
    """
    print("\n[Groq 70B] 設計と比較中...")

    short_diff = extract_minimal_diff(diff_text, max_lines=40)

    groq_context_final.append({
        "role": "user",
        "content": f"""## Ollamaの修正diff

```
{short_diff}
```

## Groq 8Bの指摘

{check_result}

最初に渡した設計・提案と照らし合わせて、この修正は正しいですか？

特に以下を確認してください:
- 設計と無関係な変更がないか
- テストだけを通すための雑な修正ではないか
- 既存機能を壊していないか
- 修正範囲が最小限か

- 正しい: 「COMPLETE」とだけ答えてください
- 問題あり: 「NG」と答えて問題点を簡潔に説明してください
"""
})
    # 👇 ここに入れる（重要）
    MAX_HISTORY = 6
    del groq_context_final[:-MAX_HISTORY]  

    response = call_groq(groq_context_final, GROQ_MODEL_FINAL)

    groq_context_final.append({"role": "assistant", "content": response})
    del groq_context_final[:-MAX_HISTORY]

    print(f"  -> 70B判断: {response[:80]}...")

    if "COMPLETE" in response.upper():
        print("  -> Groq 70B: COMPLETE！")
        return True, modified_code

    print("  -> Groq 70B: NG、Ollamaに差し戻し")
    return False, response

# ============================
# メイン処理
# ============================

def main():
    print("=" * 50)
    print("  Ollama + Groq コード修正パイプライン")
    print("=" * 50)

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="修正対象ファイルのパス")
    parser.add_argument("--design", help="設計・提案の内容")
    args = parser.parse_args()

    # ファイルパス取得
    file_path = args.file
    if not file_path:
        file_path = input("\n修正対象ファイルのパスを入力してください: ").strip()

    if not os.path.exists(file_path):
        print(f"エラー: ファイルが見つかりません: {file_path}")
        sys.exit(1)

    # 設計・提案取得
    design = args.design
    if not design:
        print("\n設計・提案を入力してください（空行1回で完了）:")
        lines = []
        empty_count = 0
        while empty_count < 1:
            line = input()
            if line == "":
                empty_count += 1
            else:
                empty_count = 0
                lines.append(line)
        design = "\n".join(lines)

    if not design.strip():
        print("エラー: 設計・提案が空です")
        sys.exit(1)

    # ファイル読み込み
    original_code = read_file(file_path)
    filename = os.path.basename(file_path)
    print(f"\n対象ファイル: {filename} ({len(original_code.splitlines())}行)")
    print(f"設計: {design[:80]}...")

    # Groq(70B)のコンテキスト初期化（設計を最初の1回だけ渡す）
    groq_context_final = [
        {
            "role": "system",
            "content": "あなたはコードレビューの専門家です。設計・提案に基づいてコードの正確性を最終判断します。"
        },
        {
            "role": "user",
            "content": f"## 設計・提案（これを常に参照してください）\n{design}\n\nこの設計・提案を記憶しておいてください。以降の最終確認はこれを基準にします。"
        }
    ]
    groq_init_response = call_groq(groq_context_final, GROQ_MODEL_FINAL)
    groq_context_final.append({"role": "assistant", "content": groq_init_response})
    print(f"\n[Groq 70B] 設計を受け取りました: {groq_init_response[:60]}...")

    # 設計をOllamaで分割
    print("\n  -> Groq待機中...")

    design_tasks = split_design(design)

    # パイプライン実行
    current_code = original_code

    for task_num, task_design in enumerate(design_tasks):
        if len(design_tasks) > 1:
            print(f"\n{'='*50}")
            print(f"  タスク {task_num+1}/{len(design_tasks)}: {task_design}")
            print(f"{'='*50}")

        current_code = read_file(file_path)

        MAX_PIPELINE_ITER = 5
        iteration = 0
        print("\n[Groq 8B] TDD用pytestを先に生成します...")

        test_code = groq_generate_pytest(current_code)

        for _ in range(2):
            if not is_test_weak(test_code):
                break

            weakness = explain_test_weakness(test_code)

            print(f"⚠️ テストが弱い → 再生成\n{weakness}")

            test_code = groq_generate_pytest(
                current_code
                + "\n\n# 以下の不足を必ず改善してください:\n"
                + weakness
                + "\n# 異常系・境界値・正常系を含めてください。"
            )

        with open("test_auto.py", "w", encoding="utf-8") as f:
            f.write(test_code)

        print("\n[pytest] 先にテストを実行します（失敗してOK）")
        test_ok, test_error = run_pytest()

        if not test_ok:
            structured_error = parse_pytest_error_to_json(test_error)
            structured_error = optimize_structured_error(structured_error)
            task_design = (
                f"[TDD pytest失敗]\n{structured_error}\n\n"
                f"このpytestを通るようにコードを修正してください。"
            )
        else:
            print("⚠️ 生成したテストが最初から通りました。テストが弱い可能性があります。")
        
        

        MAX_8B_CALLS = 3
        MAX_70B_CALLS = 2
        ai_8b_calls = 0
        ai_70b_calls = 0
        while iteration < MAX_PIPELINE_ITER:
            iteration += 1
            print(f"\n{'='*50}")
            print(f"  パイプライン実行 {iteration}回目")
            print(f"{'='*50}")

            diff_text, modified_code = ollama_generate_diff(task_design, current_code, filename)
            if diff_text is None:
               print("\n❌ Ollamaが修正できませんでした。設計をより具体的にしてください。")
               break

            if not should_run_ai(diff_text):
                print("  -> 小変更のためAIチェックをスキップ")
                is_ok = True
                check_result = "小変更のため8Bチェック省略"
            else:
                if ai_8b_calls >= MAX_8B_CALLS:
                    print("  -> 8Bチェック上限到達、AIチェックをスキップ")
                    is_ok = True
                    check_result = "8Bチェック上限到達"
                else:
                    ai_8b_calls += 1
                    is_ok, check_result = groq_check_diff(diff_text)
                    if not is_ok:
                        print("\n❌ 8BチェックNG → Ollamaへ戻す")

                        task_design = (
                            f"{task_design}\n\n"
                            f"[8B指摘]\n{check_result}\n\n"
                            f"上記の指摘に関係する箇所だけを修正してください。\n"
                            f"無関係なコードは変更しないでください。"
                        )
                        continue  # ← ここで70Bスキップ

            if is_ok:

                print("\n  -> Groq 70B待機中...")
                
                syntax_ok, syntax_error = check_python_syntax(modified_code)

                if not syntax_ok:
                    print(f"\n❌ Python構文エラー: {syntax_error}")
                    task_design = f"{task_design}\n\n[Python構文エラー]\n{syntax_error}"
                    continue

                if ai_70b_calls >= MAX_70B_CALLS:
                    print("  -> 70Bチェック上限到達、現在の修正で進む")
                    is_complete = True
                    final_result = modified_code
                else:
                    ai_70b_calls += 1
                    
                    reason = check_result if check_result else "8Bチェック省略"

                    is_complete, final_result = groq_verify_final(
                        diff_text,
                        reason,
                        modified_code,
                        groq_context_final
                    )


                if is_complete:
                    print("\n✅ 70B確認OK → pytest自動生成")

                    # 👇 ここが追加（70B後）
                    test_code = groq_generate_pytest(final_result)

                    
                    for _ in range(2):
                        if not is_test_weak(test_code):
                            break
                        weakness = explain_test_weakness(test_code)

                        print(f"⚠️ テストが弱い → 再生成\n{weakness}")

                        test_code = groq_generate_pytest(
                            final_result
                            + "\n\n# 以下の不足を必ず改善してください:\n"
                            + weakness
                            + "\n# 異常系・境界値・正常系を含めてください。"
                        )

                    with open("test_auto.py", "w", encoding="utf-8") as f:
                        f.write(test_code)

                    print("\n✅ pytest生成OK → テスト実行")

                    test_ok, test_output = run_pytest()

                    coverage = extract_coverage(test_output)

                    # ① まずテスト失敗処理
                    if not test_ok:
                        structured_error = parse_pytest_error_to_json(test_output)
                        structured_error = optimize_structured_error(structured_error)

                        print(f"\n❌ テスト失敗:\n{structured_error}")

                        task_design = (
                            f"[pytest失敗要約]\n{structured_error}\n\n"
                            f"上記のpytest失敗を直すように修正してください。"
                        )
                        continue

                     # ② そのあとカバレッジ
                    
                    if coverage < 80:
                        print(f"⚠️ カバレッジ低い: {coverage}%")

                        task_design = f"""
現在のカバレッジは{coverage}%です。
80%以上になるようにテストを改善または追加してください。
"""
                        continue

                    print("\n✅ テストOK！タスク完了！")
                    write_file(file_path, final_result)
                    current_code = final_result
                    break
                else:
                    print(f"\n❌ 70Bの指摘: {final_result[:80]}...")
                    task_design = (
                        f"{task_design}\n\n"
                        f"[70B指摘]\n{final_result}\n\n"
                        f"上記の指摘に関係する箇所だけを修正してください。\n"
                        f"無関係なコードは変更しないでください。"
                    )


        else:
            print("❌ 最大試行回数に到達しました。タスク失敗")

    print(f"\n✅ 完了！ファイルを更新しました: {file_path}")
    # 最終バックアップ
    if not os.getenv("CI"):
        backup_path = file_path + ".backup"
        write_file(backup_path, original_code)
        print(f"📁 バックアップ: {backup_path}")

    final_diff = compute_diff(original_code, current_code, filename)
    print(f"\n## 最終的な変更内容\n{final_diff}")

if __name__ == "__main__":
    main()