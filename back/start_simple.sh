#!/bin/sh
export BACKEND=simple
uvicorn main:app --reload
