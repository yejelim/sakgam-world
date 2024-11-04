# 베이스 이미지 설정
FROM python:3.9

# 작업 디렉토리 설정
WORKDIR /app

# 로컬 파일을 컨테이너에 복사
COPY . /app

# 필요한 패키지 설치
RUN pip install -r requirements.txt

# 애플리케이션 실행 명령어 설정
CMD ["streamlit", "run", "app.py", "--server.port=8000", "--server.address=0.0.0.0"]
