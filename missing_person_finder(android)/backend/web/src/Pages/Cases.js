import React,{useContext, useEffect, useState} from 'react';
import Card from 'react-bootstrap/Card';
import firebaseApp from '../Firebase';
import 'firebase/compat/database';
import {useNavigate} from 'react-router-dom';
import { Cardcontext } from '../contexts/CardsContext';
import { Searchcontext } from '../contexts/SearchContext';
const Cases = () => {
    const[snaps,setSnaps]=useState([])
    let array=[]
    const navigate=useNavigate();
    const {setCard}=useContext(Cardcontext)
    const{item}=useContext(Searchcontext)
    const getCardDetails=(index)=>{
        setCard(snaps[index])
        navigate(`case/${index}`)
    }
    const getdata=()=>{
        firebaseApp.database().ref('server').child('missing data')
            .once('value').then((snapshot)=>{
                snapshot.forEach(element=>{
                    array.push(element.val())
                })  
                setSnaps(array)
            }) 
    }
    useEffect(() => {
         const getDetails=async()=>{
              await getdata()
              
          }
          getDetails();
          
    },[snaps])

    return (
        <div className='row' style={{ display:'flex'}}>
           <h5>Results : {snaps.length} items</h5>
           {  item.length>1?
               snaps.filter((val)=>{
                 if(val.firstname.toLowerCase().includes(item.toLowerCase())){
                   return val
                 }
                
               })
               .map((missdata)=>{
                   return <Card onClick={()=>getCardDetails(snaps.indexOf(missdata)) } style={{ margin:"2rem", width: '15rem' }}>
                   <Card.Img variant="top" src={missdata.imageUrl} height='250px'/>
                   <Card.Body>
                     <Card.Title style={{'text-align':'center'}}>{missdata.firstname + ' '+missdata.lastname}</Card.Title>
                   </Card.Body>
                 </Card>
               }):
               snaps.map((missdata,index)=>{
                return <Card onClick={()=>getCardDetails(index) } style={{ margin:"2rem", width: '15rem' }}>
                <Card.Img variant="top" src={missdata.imageUrl} height='250px'/>
                <Card.Body>
                  <Card.Title style={{'text-align':'center'}}>{missdata.firstname + ' '+missdata.lastname}</Card.Title>
                </Card.Body>
              </Card>
            })

           }
        </div>
    );
}

export default Cases;
