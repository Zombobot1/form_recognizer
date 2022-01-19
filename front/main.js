import './style.css';

import { saveAs } from 'file-saver';

const api =
  process.env.NODE_ENV === 'development'
    ? 'http://localhost:8000'
    : location.origin;

let canClick = true;

const docxType =
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document';

async function sleep(ms) {
  await new Promise((resolve) => setTimeout(resolve, ms));
  if (failInput.checked) throw new Error('Parsing failed');
}

async function stubParseForm() {
  await sleep(2000);
  return new File(['Hello, world!'], 'parsed.docx', { type: docxType });
}

async function parseForm(file) {
  // return stubParseForm()

  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(api + '/parseForm', {
    method: 'POST',
    body: formData,
  });

  const blob = await response.blob();

  return new File([blob], 'parsed.docx', { type: docxType });
}

async function processFile(file) {
  label.textContent = `Processing: ${file.name}`;
  ripple.classList.toggle('show');
  dropContainer.classList.toggle('disable');

  canClick = false;

  try {
    const parsedFile = await parseForm(file);

    dropContainer.classList.toggle('disable');
    ripple.classList.toggle('show');
    label.textContent = onLeaveText;
    fileInput.value = '';

    canClick = true;
    saveAs(parsedFile);
  } catch (e) {
    label.classList.toggle('hide');
    retry.classList.toggle('show');
    ripple.classList.toggle('show');
    error.textContent = `Error: ${e.message}`;
    fileInput.value = ''; // clean value when fail is processed
  }
}

const onEnterText = 'Drop file';
const onLeaveText = 'Drop file here or click to select';

const dropContainer = document.querySelector('.drop-container');
const label = document.querySelector('.drop-label-text');
const error = document.querySelector('.drop-label-error');
const fileInput = document.querySelector('.file-input');
const failInput = document.querySelector('.fail-checkbox');
const ripple = document.querySelector('.spinner');
const retry = document.querySelector('.retry-btn');

const onDragOver = (e) => e.preventDefault();
const onDragEnter = () => (label.textContent = onEnterText);
const onDragLeave = () => (label.textContent = onLeaveText);

function onDrop(e) {
  e.preventDefault();
  processFile(e.dataTransfer.files[0]);
}

const onClick = () => {
  if (canClick) fileInput.click();
};

dropContainer.addEventListener('dragover', onDragOver);
dropContainer.addEventListener('dragenter', onDragEnter);
dropContainer.addEventListener('dragleave', onDragLeave);
dropContainer.addEventListener('drop', onDrop);

dropContainer.addEventListener('click', onClick);

function onFileChange() {
  if (fileInput.files.length) processFile(fileInput.files[0]);
}

fileInput.addEventListener('change', onFileChange);

function onRetryClick(e) {
  e.stopPropagation();
  label.classList.toggle('hide');
  retry.classList.toggle('show');
  error.textContent = ``;
  dropContainer.classList.toggle('disable');
  label.textContent = onLeaveText;

  canClick = true;
}

retry.addEventListener('click', onRetryClick);
