import Register from "./Pages/Register";
import {BrowserRouter as Router, Route, Routes} from 'react-router-dom';
import './App.css';
import Cases from "./Pages/Cases";
function App() {
  return (
    <div className="App">
      <div className="Navbar">
        <a href="#">Cases</a>
        <a  href="/register">Register</a>
        <a  href="#">About</a>
      </div>
    <Router>
      <Routes>
      <Route exact path="/" element={<Register/>}></Route>
      </Routes>
    </Router>
    </div>
  );
}

export default App;
