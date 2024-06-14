import './App.css';
import { Button, Form, Spinner, Dropdown} from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';
import React, { useState, useEffect } from "react";
import io from "socket.io-client";

const socket = io('http://localhost:5000');


function App() {

  const [formData, setFormData] = useState({
    text: '',
    file: null
  });

  const [submitClicked, setSubmitClicked] = useState(false);
  const [dashboardAvailable, setDashboardAvailable] = useState(false);
  const [iframeKey, setIframeKey] = useState(0);
  const [selectedExplanation, setSelectedExplanation] = useState('Text');
  const [selectedModel, setSelectedModel] = useState('');

  
  
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
        ...formData,
        [name]: value
      });
  };

  const handleSubmit = async (e) => {
      setSubmitClicked(true);
      setDashboardAvailable(false);
      e.preventDefault();

      const form = new FormData();
      if (selectedExplanation === 'Text') {
        form.append('text', formData.text);
      } else if (formData.file) {
        form.append('file', formData.file);
      }
      form.append('selected_model', selectedModel); // Add selected model to the form data

      let link = 'http://localhost:5000/explain_';
      switch (selectedExplanation){
        case 'Text':
          link += "text"
          break;
        case 'Photo':
          link += "image"
          break;
        case 'Time Series':
          link += "time-series"
          break;
        default:
          link += ''
          break;
      }      
      const response = await fetch(link , {
          method: 'POST',
          body: form
      });
      const data = await response.json();
      console.log(data);
  };

  
  // Function to handle file upload
  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setFormData({ file });
    }
  };

  const handleExplanationChange = (explanation) => {
    setSelectedExplanation(explanation);
    setFormData({ text: '', file: null });
    setSelectedModel('');
  }

  const handleModelChange = (model) => {
    setSelectedModel(model);
  };

  useEffect(() => {
    socket.on('dashboard_status', (data) => {
      setTimeout(() => {
        setDashboardAvailable(data.running);

        setIframeKey((prevKey) => prevKey + 1);
      }, 6000); // Adjust the timeout as needed
    });
  }, []);

  return (
    <div className="App">
      <h1 className="mb-4">XAI Explanations Generator</h1>
      <Form onSubmit={handleSubmit}>
        <Form.Group className="mb-3 form-inline" controlId="formInput">
        <div className="d-flex align-items-center">
          <Dropdown className='ml-2 mb-2 mr-4 dropdown-explanation'>
              <Dropdown.Toggle variant="primary" id="dropdown-explanation">
                {selectedExplanation} Explanation
              </Dropdown.Toggle>
              <Dropdown.Menu>
                <Dropdown.Item onClick={() => handleExplanationChange('Text')}>Text</Dropdown.Item>
                <Dropdown.Item onClick={() => handleExplanationChange('Photo')}>Photo</Dropdown.Item>
                <Dropdown.Item onClick={() => handleExplanationChange('Time Series')}>Time Series</Dropdown.Item>
              </Dropdown.Menu>
          </Dropdown>
          {selectedExplanation === 'Text' && (
                          <Dropdown className='mb-2 ml-2 dropdown-model'>
                              <Dropdown.Toggle variant="primary" id="dropdown-model">
                                  {selectedModel ? selectedModel : 'Choose Model'}
                              </Dropdown.Toggle>
                              <Dropdown.Menu>
                                  <Dropdown.Item onClick={() => handleModelChange('CNN')}>CNN</Dropdown.Item>
                                  <Dropdown.Item onClick={() => handleModelChange('Roberta')}>RoBERTa</Dropdown.Item>
                              </Dropdown.Menu>
                          </Dropdown>
                      )}
                      {selectedExplanation === 'Photo' && (
                          <Dropdown className='mb-2 ml-2 dropdown-model'>
                              <Dropdown.Toggle variant="primary" id="dropdown-model">
                                  {selectedModel ? selectedModel : 'Choose Model'}
                              </Dropdown.Toggle>
                              <Dropdown.Menu>
                                  <Dropdown.Item onClick={() => handleModelChange('resnet50')}>ResNet 50</Dropdown.Item>
                                  <Dropdown.Item onClick={() => handleModelChange('alexnet')}>AlexNet</Dropdown.Item>
                              </Dropdown.Menu>
                          </Dropdown>
                      )}
                      {selectedExplanation === 'Time Series' && (
                          <Dropdown className='mb-2 ml-2 dropdown-model'>
                              <Dropdown.Toggle variant="primary" id="dropdown-model">
                                  {selectedModel ? selectedModel : 'Choose Model'}
                              </Dropdown.Toggle>
                              <Dropdown.Menu>
                                  <Dropdown.Item onClick={() => handleModelChange('Seasonal Decompose')}>Seasonal Decompose</Dropdown.Item>
                                  <Dropdown.Item onClick={() => handleModelChange('SVM')}>One Class SVM</Dropdown.Item>
                              </Dropdown.Menu>
                          </Dropdown>
                      )}
        </div>
        {selectedExplanation === 'Text' && (
          <Form.Control type= "text" name="text" value={formData.text} placeholder="Inserisci input da spiegare..." onChange={handleChange} className="custom-input" />
        )}
        {selectedExplanation === 'Photo' && (
            <Form.Control
              name="file"
              type="file"
              onChange={handleFileUpload}
              accept=".png, .jpg"
              label="Upload PNG file"
              className="ml-2"
            />
        )}
        {selectedExplanation === 'Time Series' && (
            <Form.Control
              name="file"
              type="file"
              onChange={handleFileUpload}
              accept=".csv"
              label="Upload CSV file"
              className="ml-2"
            />
          )}
          <Button className="mt-2" variant="primary" type="submit" disabled={!selectedModel}>
          Submit
          </Button>
        </Form.Group>
      </Form>
      {submitClicked && (
      <iframe
        key={iframeKey} // Add a unique key to force iframe reload
        src="http://localhost:8050/"
        title="omnixai"
        style={{ display: dashboardAvailable ? 'block' : 'none' }}>
      </iframe>
      )}
      {submitClicked && !dashboardAvailable && <Spinner animation="border" variant="primary" />}
    </div>
  );
}


export default App;