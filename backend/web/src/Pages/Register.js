import React, { useState } from 'react';
import './Register.css';
import axios from 'axios'
import {RegionDropdown} from 'react-country-region-selector';
function Register() {
    const [fname,SetFname]=useState('');
    const [Lname,SetLname]=useState('');
    const [Country,SetCountry]=useState('');
    const [des,SetDes]=useState('');
    const [file,setFile] = useState(null)
    const imagehandle =(e)=>{
        const datas = e.currentTarget.files[0]
        setFile(datas)
    }
    const handleSubmit= e =>{
        e.preventDefault()
        console.log(file)
        let form_data = new FormData();
        form_data.append('firstname',fname);
        form_data.append('lastname',Lname);
        form_data.append('image', file);
        form_data.append('country', Country);
        form_data.append('description', des);
            axios.post('register/', form_data, {
            headers: {
                'content-type': 'multipart/form-data'
            }
            })
        
        .then((res)=>{
            console.log(res.data)
        })
        .catch((err)=>console.log(err))
        
        
    }

        return (
            <div class="container">
            <h1>Register New case</h1>
        <form onSubmit={handleSubmit} autoComplete='off'>
            <div class="row">
            <div class="col-25">
                <label for="fname">First Name</label>
            </div>
            <div class="col-75">
                <input type="text" value={fname} onChange={e=> SetFname(e.currentTarget.value)} id="fname" name="firstname" placeholder="Your name.." />
            </div>
            </div>
            <div class="row">
            <div class="col-25">
                <label for="lname">Last Name</label>
            </div>
            <div class="col-75">
                <input type="text" value={Lname} onChange={e=> SetLname(e.currentTarget.value)} id="lname" name="lastname" placeholder="Your last name.."/>
            </div>
            </div>
            <div class="row">
            <div class="col-25">
                <label for="country">Last Seen</label>
            </div>
            <div class="col-75">
                <RegionDropdown country="India" value={Country} name="place" onChange={e=> SetCountry(e)}/>

            </div>
            </div>
            <div class="row">
            <div class="col-25">
                <label for="description">Description</label>
            </div>
            <div class="col-75">
                <textarea id="des" value={des} onChange={e=> SetDes(e.currentTarget.value)} name="description" placeholder="Write something.."></textarea>
            </div>
            <div class="col-25">
                <label for="Image">ImageUpload</label>
            </div>
            <div class="col-75">
                <input type="file" name='img' src={file} onChange={imagehandle} multiple accept="image/*"/>
            </div>
            </div>
            <div class="row">
            <input type="submit" value="Submit"/>
            </div>
        </form>
    </div>
        );
    
}

export default Register;
