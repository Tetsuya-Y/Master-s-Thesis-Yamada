import MeCab
import ipadic
import glob
import os
import re

# ================= 設定エリア =================
# 入力ファイルがあるフォルダ
INPUT_FOLDER = "input_txt"
# 出力ファイルを保存するフォルダ
OUTPUT_FOLDER = "output_txt"

# 残したい品詞
TARGET_POS = ["名詞", "動詞", "形容詞"]
# ============================================

def setup_mecab():
    """MeCabの初期化（ipadicを明示的に指定）"""
    return MeCab.Tagger(ipadic.MECAB_ARGS)

def clean_text(text):
    """
    ノイズ除去関数
    半角アルファベット(a-z, A-Z) と 半角記号(!-~等) を削除します。
    数字(0-9)は残す設定にしていますが、消したい場合は正規表現に 0-9 を追加してください。
    """
    # 半角英字と半角記号を空文字に置換
    # [a-zA-Z]: アルファベット
    # [!-~]: 半角記号（!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~）
    cleaned = re.sub(r'[a-zA-Z!-~]', '', text)
    return cleaned

def process_file(file_path, tagger):
    """1つのファイルを処理して結果のテキストを返す"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    processed_lines = []

    for line in lines:
        # 1. ノイズ除去（MeCabにかける前に不要な文字を消す）
        line_clean = clean_text(line)
        
        # 2. 形態素解析
        # parseToNodeを使ってノードごとに処理
        node = tagger.parseToNode(line_clean)
        
        words = []
        while node:
            # node.feature は "品詞,品詞細分類1,品詞細分類2,...,原形,..." の文字列
            features = node.feature.split(',')
            
            # featuresの要素数が少ない場合（BOS/EOSなど）はスキップ
            if len(features) < 1:
                node = node.next
                continue
            
            pos = features[0] # 品詞（名詞、動詞など）

            # 指定した品詞のみ抽出
            if pos in TARGET_POS:
                # node.surface: 表層形（原文のまま）
                # node.featureから原形を取りたい場合は features[6] を使う
                # ここでは「テキストから抜き出す」意図を汲んで表層形(surface)を使用
                words.append(node.surface)

            node = node.next
        
        # 抽出した単語をスペース区切りで連結し、元の改行コードを付与
        # 単語が1つもない行でも、空行として改行を残す設定
        processed_lines.append(" ".join(words) + "\n")

    return "".join(processed_lines)

def main():
    # MeCabの準備
    try:
        tagger = setup_mecab()
    except Exception as e:
        print(f"エラー: MeCabの初期化に失敗しました。\n詳細: {e}")
        return

    # 出力フォルダがなければ作成
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"フォルダを作成しました: {OUTPUT_FOLDER}")

    # 指定フォルダ内のtxtファイルを取得
    input_files = glob.glob(os.path.join(INPUT_FOLDER, "*.txt"))
    
    if not input_files:
        print(f"警告: '{INPUT_FOLDER}' フォルダ内に .txt ファイルが見つかりません。")
        return

    print(f"{len(input_files)} 個のファイルを処理します...")

    for file_path in input_files:
        try:
            # ファイル名を取得（パスを除く）
            filename = os.path.basename(file_path)
            output_path = os.path.join(OUTPUT_FOLDER, f"processed_{filename}")
            
            # 処理実行
            result_text = process_file(file_path, tagger)
            
            # 書き込み
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result_text)
                
            print(f"完了: {filename} -> {output_path}")
            
        except Exception as e:
            print(f"失敗: {file_path} の処理中にエラーが発生しました。\n詳細: {e}")

    print("\nすべての処理が完了しました。")

if __name__ == "__main__":
    main()