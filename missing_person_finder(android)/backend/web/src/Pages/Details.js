import React, { useContext } from 'react';
import Header from '../components/Header';
import { Cardcontext } from '../contexts/CardsContext';
import './Details.css';
const Details = () => {
    const {card}=useContext(Cardcontext)
    return (
        
        <div className="container-center-horizontal">
        {
        card? <div className="desktop-1screen">
        
         <Header 
         Name={card.firstname+' '+card.lastname}
         Gender={card.gender}
         Age={card.age}
         imageUrl={card.imageUrl}
         report={card.reportingdate}
         region={card.region}

        />
          <div className="overlap-group1 firacode-normal-black-36px">
            <div className="description">
              Description :
            </div>
            <div className="add-content" style={{backgroundColor:'#AD8A8A'}}>
              {card.description}
            </div>
          </div> 
        </div>:window.location.replace('/')
          }
      </div>
        
    );
}

export default Details;
