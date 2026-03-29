import sys

file_path = r'E:\Desenvolvimento\Python\tcc\tts-finetunning-8gb-vram\.venv\Lib\site-packages\speechbrain\utils\fetching.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

target = """    except HTTPError as e:
        if "404 Client Error" in str(e):
            raise ValueError("File not found on HF hub") from e
        raise"""

replacement = """    except HTTPError as e:
        if "404 Client Error" in str(e):
            raise ValueError("File not found on HF hub") from e
        raise
    except Exception as e:
        if "404" in str(e) or "Not Found" in str(e) or "Entry Not Found" in str(e):
            raise ValueError("File not found on HF hub") from e
        raise"""

if target in content:
    content = content.replace(target, replacement)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print('PATCH_SUCCESS')
else:
    if replacement in content:
        print('ALREADY_PATCHED')
    else:
        print('TARGET_NOT_FOUND')
