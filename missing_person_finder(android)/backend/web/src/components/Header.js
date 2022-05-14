import React from 'react';
import './Header.css';
const Header = (props) => {
    return (
        <div className="rectangle-1">
        <img className="ellipse-1 " src={props.url} />
        <div className="flex-col">
          <h1 className="title">
            Name : {props.Name}
          </h1>
          <div className="age">
            Gender : {props.Gender}
          </div>
          <div className="age">
            Age : {props.Age}
          </div>
          <div className="overlap-group firacode-bold-black-23-5px">
            <div className="reporting-time-and-date">
              Reportin Date : {props.report}
            </div>
            <div className="last-seen">
              Region : {props.region}
            </div>
          </div>
        </div>
      </div>
    );
}

export default Header;
