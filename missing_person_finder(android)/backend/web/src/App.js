import Register from "./Pages/Register";
import React, {useState} from 'react';
import {  Route, Routes,useNavigate} from 'react-router-dom';
import {Form,FormControl,Nav,Container,Navbar} from 'react-bootstrap';
import './App.css';
import Cases from "./Pages/Cases";
import Match from "./Pages/Match";

import {ToastContainer} from 'react-toastify'
import 'react-toastify/dist/ReactToastify.css'
import Details from "./Pages/Details";
import { Searchcontext } from "./contexts/SearchContext";
function App() {
  
  const navigate=useNavigate();
  const [item, setItem] = useState("");
  return (
    
    <div className="App">
      <Navbar bg="light" expand="lg">
        <Container fluid>
          <Navbar.Brand >Missing Person Finder</Navbar.Brand>
          <Navbar.Toggle aria-controls="navbarScroll" />
          <Navbar.Collapse id="navbarScroll">
            <Nav
              className="me-auto my-2 my-lg-0"
              style={{ maxHeight: '100px' }}
              navbarScroll
            >
              <Nav.Link onClick={()=>navigate("/")}>View Cases</Nav.Link>
              <Nav.Link onClick={()=>navigate("register_case/")}>Register Case</Nav.Link>
              <Nav.Link onClick={()=>navigate("match_cases/")}>Match Cases</Nav.Link>
            </Nav>
            <Form className="d-flex">
              <FormControl
                type="search"
                placeholder="Search"
                className="me-2"
                aria-label="Search"
                onChange={(e)=>setItem(e.target.value)}
              />
            </Form>
          </Navbar.Collapse>
        </Container>
      </Navbar>
    
      <Searchcontext.Provider value={{item}}>
      <Routes>
      <Route exact path="/" element={<Cases/>}></Route>
      <Route  path="register_case/" element={<Register/>}></Route>
      <Route  path="match_cases/" element={<Match/>}></Route>
      <Route path='case/:id' element={<Details />}></Route>
      </Routes>
      </Searchcontext.Provider>
    
      <ToastContainer autoClose={10000}/>
    </div>
  
  );
}

export default App;
