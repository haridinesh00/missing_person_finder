import React, { useState } from 'react';
import './Register.css';
import axios from 'axios'
import firebaseApp from '../Firebase';
import 'firebase/compat/database';
import {toast} from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
function Register() {
    const [fname,SetFname]=useState('');
    const [Lname,SetLname]=useState('');
    const [age, SetAge] = useState(null);
    const [gender, setGender] = useState('');
    const [region,SetRegion]=useState('');
    const [des,SetDes]=useState('');
    const [file,setFile] = useState(null)
    const[value,setValue] =useState(0)
    let dt= new Date()
    console.log(dt.getFullYear() + "/" + (dt.getMonth() + 1) + "/" + dt.getDate())
    const imagehandle =(e)=>{
        const datas = e.currentTarget.files[0]
        setFile(datas)
        setValue(1)
    }
    const handleSubmit= e =>{
        e.preventDefault()
        value?toast.info('Waiting for Response',{autoClose:10000,position: toast.POSITION.TOP_CENTER}):toast.error("Failed in Image Upload",{autoClose:10000,position: toast.POSITION.TOP_CENTER})
        firebaseApp.storage().ref(`images/${file.name}`).put(file).then(({ref})=>{
    
            ref.getDownloadURL().then((imageUrl)=>{
                let form_data = new FormData();
                form_data.append('firstname',fname);
                form_data.append('lastname',Lname);
                form_data.append('image',file);
                    axios.post(`${value}`, form_data, {
                    headers: {
                        'content-type': 'multipart/form-data'
                    }
                    })
                
                .then((res)=>{
                    
                    var storage = firebaseApp.database().ref("server/missing data").child(fname+Lname);
                    storage.set({
                        'age':age,
                        'description':des,
                        'encoding':res.data,
                        'firstname':fname,
                        'gender':gender,
                        'lastname':Lname,
                        'region':region,
                        'reportingdate':dt.getFullYear() + "/" + (dt.getMonth() + 1) + "/" + dt.getDate(),
                        'imageUrl':imageUrl.toString()

                    })
                    .then(obj=>toast.success('Sucessfully Uploaded',{autoClose:10000,position: toast.POSITION.TOP_CENTER}))
                    .catch(err=>toast.error(`${err.message}`,{autoClose:10000,position: toast.POSITION.TOP_CENTER}))
                })
                .catch((err)=> toast.error("Invalid Image",{autoClose:10000,position: toast.POSITION.TOP_CENTER}))
            })
        })
        .catch(err=>{
            toast.error(`${err.message}`,{autoClose:10000,position: toast.POSITION.TOP_CENTER})
        })
        
        
    }

        return (
            <div className="container">
            <h1>Register New case</h1>
        <form onSubmit={handleSubmit} autoComplete='off'>
            <div className="row">
            <div className="col-25">
                <label for="fname">First Name</label>
            </div>
            <div className="col-75">
                <input type="text" value={fname} onChange={e=> SetFname(e.currentTarget.value)} id="fname" name="firstname" placeholder="Your name.." required />
            </div>
            </div>
            <div className="row">
            <div className="col-25">
                <label for="lname">Last Name</label>
            </div>
            <div className="col-75">
                <input type="text" value={Lname} onChange={e=> SetLname(e.currentTarget.value)} id="lname" name="lastname" placeholder="Your last name.." required/>
            </div>
            </div>
            <div className="row">
            <div className="col-25">
                <label for="lname">Age</label>
            </div>
            <div className="col-75">
                <input type="text" value={age} onChange={e=> SetAge(e.currentTarget.value)} id="age" name="age" placeholder="Age" required/>
            </div>
            </div>
            <div className="row">
            <div className="col-25">
                <label for="lname">Gender</label>
            </div>
            <div className="custom-select col-75" >
                <select title='Gender' onChange={(e)=>setGender(e.target.value)}  required>
                    <option selected>---select gender---</option>
                    <option>Male</option>
                    <option >Female</option>
                    <option >other</option>
                </select>
            </div>
            </div>
            <div className="row">
            <div className="col-25">
                <label for="country">Region</label>
            </div>
            <div className="col-75">
                <input type="text" value={region} onChange={e=> SetRegion(e.currentTarget.value)} id="region" name="region" placeholder="region" required/>
            </div>
            </div>
            <div className="row">
            <div className="col-25">
                <label for="description">Description</label>
            </div>
            <div className="col-75">
                <textarea id="des" value={des} onChange={e=> SetDes(e.currentTarget.value)} name="description" placeholder="Write something.." required></textarea>
            </div>
            <div className="col-25">
                <label for="Image">ImageUpload</label>
            </div>
            <div className="col-75">
                <input type="file" name='img' src={file} onChange={imagehandle} multiple accept="image/*" />
            </div>
            </div>
            <div className="row">
            <input type="submit" onClick={imagehandle} value="Submit"/>
            </div>
        </form>
    </div>
        );
    
}

export default Register;
