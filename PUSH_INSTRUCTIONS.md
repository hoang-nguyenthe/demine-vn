# DeMine-VN — Push lên GitHub và deploy Streamlit Cloud

Bản trình diễn Streamlit đã sẵn sàng tại `~/projects/demine-vn/scripts/demo_app.py`.

## 1. Chạy thử local trên Mac

```bash
cd ~/projects/demine-vn
pip install -r requirements.txt
streamlit run scripts/demo_app.py
```

## 2. Tạo repo trên GitHub

Vào https://github.com/new → tên `demine-vn` → Public → **không** tick "Add
README/.gitignore/license" → Create.

## 3. Push repo local lên GitHub

```bash
cd ~/projects/demine-vn
git remote add origin https://github.com/hoang-nguyenthe/demine-vn.git
git push -u origin main
```

Nếu remote đã tồn tại, chạy trước:
```bash
git remote remove origin
```

Xác thực dùng Personal Access Token — https://github.com/settings/tokens?type=beta →
cấp `Contents: Read and write` cho repo `demine-vn`.

## 4. Deploy Streamlit Community Cloud

Vào https://share.streamlit.io/ → **Create app** → điền:
- Repository: `hoang-nguyenthe/demine-vn`
- Branch: `main`
- Main file path: `scripts/demo_app.py`

→ **Deploy**. Sau ~2 phút app chạy tại `https://<tên>.streamlit.app/`.

## 5. Cập nhật về sau

```bash
cd ~/projects/demine-vn
git add -A
git commit -m "Mô tả thay đổi"
git push
```
