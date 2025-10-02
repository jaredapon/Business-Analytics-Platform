from fastapi import FastAPI
from minio import Minio
from minio.error import S3Error

app = FastAPI()

minio_client = Minio(
    "localhost:9000",
    access_key="booklatte",
    secret_key="password",
    secure=False
)
@app.get("/presigned-url/") 
def get_presigned_url(bucket: str, object_name: str, expires: int = 3600):
    try:
        url = minio_client.presigned_get_object(bucket, object_name, expires=expires)
        return {"url": url}
    except S3Error as e:
        return {"error": str(e)}
