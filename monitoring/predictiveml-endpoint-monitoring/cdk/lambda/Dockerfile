FROM --platform=amd64 public.ecr.aws/lambda/python:3.12

ARG HANDLER_FILE
COPY ${HANDLER_FILE}.py ${LAMBDA_TASK_ROOT}/
COPY requirements.txt ${LAMBDA_TASK_ROOT}/

RUN pip install --no-cache-dir -r requirements.txt

# Create a stable handler entry point that delegates to the actual handler.
# This sidesteps the issue of Docker CMD not expanding ARG values at runtime.
RUN printf "from %s import handler\n" "${HANDLER_FILE}" > ${LAMBDA_TASK_ROOT}/app.py

CMD ["app.handler"]
