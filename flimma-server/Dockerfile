FROM python:3.6-stretch

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /usr/src/fed_gwas/

RUN mkdir data result

COPY ./requirements.txt /usr/src/fed_gwas/requirements.txt

RUN pip3 install -r /usr/src/fed_gwas/requirements.txt
RUN pip3 install gunicorn

COPY docker-entrypoint.sh /entrypoint.sh

COPY . /usr/src/fed_gwas/

EXPOSE 8000

ENTRYPOINT ["sh", "/entrypoint.sh"]
