import Register from "./Pages/Register";
import {BrowserRouter as Router, Route, Routes} from 'react-router-dom';
import './App.css';
import Match_Cases from "./Pages/Match_Cases";
function App() {
  return (
    <div className="App">
      <div className="Navbar">
        <a href="match_cases/">Match Cases</a>
        <a  href="/">Register</a>
        <a  href="">About</a>
      </div>
    <Router>
      <Routes>
      <Route exact path="/" element={<Register/>}></Route>
      <Route path="match_cases/" element={<Match_Cases />}></Route>
      </Routes>
    </Router>
    </div>
  );
}

export default App;
