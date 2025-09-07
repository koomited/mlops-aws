from public.ecr.aws/lambda/python:3.13

RUN pip install -U pip


COPY requirements.txt .
RUN pip install -r requirements.txt

COPY lambda_function.py .

CMD ["lambda_function.lambda_handler"]