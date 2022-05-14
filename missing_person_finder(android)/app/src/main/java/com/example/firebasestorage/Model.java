package com.example.firebasestorage;

public class Model {
    private String imageUrl;
    private String firstname;
    private String lastname;
    private String country;
    private String description;
    public Model(){

    }
    public Model(String imageUrl, String firstname, String lastname, String country, String description){
        this.imageUrl = imageUrl;
        this.firstname = firstname;
        this.lastname = lastname;
        this.country = country;
        this.description = description;
    }

    public String getImageUrl() {
        return imageUrl;
    }

    public String getFirstName() { return firstname; }

    public String getLastName() { return lastname; }

    public String getCountry() { return country; }

    public String getDescription() { return description; }

    public void setImageUrl(String imageUrl) {
        this.imageUrl = imageUrl;
    }
}
