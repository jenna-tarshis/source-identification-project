#!/bin/bash
set -e
export PGPASSWORD=$POSTGRES_PASSWORD;
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
  CREATE DATABASE airflow;  -- for airflow logistics
  CREATE USER $APP_DB_USER WITH PASSWORD '$APP_DB_PASS';
  CREATE DATABASE $APP_DB_NAME;  -- for actual sims app data
  GRANT ALL PRIVILEGES ON DATABASE $APP_DB_NAME TO $APP_DB_USER;
  \connect $APP_DB_NAME $APP_DB_USER
  BEGIN;
    /*  EXTENSIONS  */

    CREATE EXTENSION IF NOT EXISTS tablefunc;

    /*  SCHEMA matching sims-prod  */

    CREATE SCHEMA IF NOT EXISTS partitions;

    /* --------------- Sample Ingestion --------------- */

    CREATE TABLE IF NOT EXISTS raw_samples (
        id SERIAL PRIMARY KEY,
        received_at BIGINT NOT NULL,
        processed_at BIGINT,
        sample_string_jsonb JSONB NOT NULL
    );

    CREATE TABLE IF NOT EXISTS raw_samples_packet_id (
        id SERIAL PRIMARY KEY,
        received_at BIGINT NOT NULL,
        processed_at BIGINT,
        sample_string_jsonb JSONB NOT NULL
    );


    CREATE TABLE IF NOT EXISTS raw_samples_pmd (
        id SERIAL PRIMARY KEY,
        processed_at BIGINT,
        ble_name varchar(50),
        rssi INT,
        sampled_at bigint,
        gateway_mac VARCHAR(50),
        received_at BIGINT NOT NULL,
        gateway_info JSONB NOT NULL,   -- original json that includes GatewayMAC and TimeStamp
        original_data JSONB NOT NULL,  -- original json that includes PMD data packet (TimeStamp, BLEMAC, RSSI, etc.)
        parsed_data JSONB NOT NULL  -- parsed data: sensor1, sensor2, humidity temperature, pressure, company_id
    );

    CREATE TABLE IF NOT EXISTS pmd_info
                               (
                                   id SERIAL,
                                   ble_name    varchar(50) NOT NULL,
                                   rssi        integer NOT NULL,
                                   sampled_at  bigint NOT NULL,
                                   gateway_mac varchar(50) NOT NULL,
                                   primary key (sampled_at, ble_name, rssi, gateway_mac)
                               )partition by RANGE (sampled_at);

    CREATE TABLE IF NOT EXISTS bad_samples (
        id SERIAL PRIMARY KEY,
        raw_samples_id INT NOT NULL,
        -- 1 for samples from raw_samples_packet_id, 0 for those from raw_samples (from migration dag)
        sample_source  SMALLINT,
        received_at BIGINT NOT NULL,
        processed_at BIGINT NOT NULL,
        sample_string_jsonb JSONB NOT NULL,
        log_message text
    );

    CREATE TABLE IF NOT EXISTS processed_samples (
        id SERIAL NOT NULL,
        raw_samples_id INT NOT NULL,
        -- 1 for samples from raw_samples_packet_id, 0 for those from raw_samples (from migration dag)
        sample_source  SMALLINT,
        sampled_at BIGINT NOT NULL,
        received_at BIGINT NOT NULL,
        processed_at BIGINT NOT NULL,
        equipment_id INT NOT NULL,
        lat FLOAT8,
        lon FLOAT8,
        alt FLOAT8,
        location_unknown bool,
        sensor_id INT NOT NULL,
        sensor_value FLOAT4 NOT NULL,
        raw_sensor_value FLOAT4,  -- optional field if sample was passed in as raw
        CONSTRAINT processed_samples_pkey PRIMARY KEY (sampled_at, sensor_id)
    ) PARTITION by range (sampled_at);

    CREATE TABLE IF NOT EXISTS average_interval_samples (
            id SERIAL NOT NULL,
            interval INT NOT NULL,
            sampled_at BIGINT NOT NULL,
            equipment_id INT NOT NULL,
            lat FLOAT8,
            lon FLOAT8,
            sensor_id INT NOT NULL,
            raw_sensor_value FLOAT4,  -- optional field if sample was passed in as raw
            sensor_value FLOAT4 NOT NULL
        ) PARTITION by range (sampled_at);

    CREATE TABLE IF NOT EXISTS average_interval_virtual_samples (
                id SERIAL NOT NULL,
                interval INT NOT NULL,
                sampled_at BIGINT NOT NULL,
                virtual_equipment_id INT,
                lat FLOAT8,
                lon FLOAT8,
                virtual_sensor_id INT,
                sensor_value FLOAT4,
                formula_id INT,
                real_equipment_id INT
            ) PARTITION by range (sampled_at);

    CREATE TABLE IF NOT EXISTS aqi_values (
                id SERIAL NOT NULL,
                sampled_at BIGINT NOT NULL,
                sensor_id INT NOT NULL,
                aqi_value FLOAT4 NOT NULL,
                aqi_percent FLOAT4 NOT NULL,
                aqi_zone VARCHAR(25) NOT NULL
            ) PARTITION by range (sampled_at);

    CREATE TABLE IF NOT EXISTS virtual_aqi_values (
                    id SERIAL NOT NULL,
                    sampled_at BIGINT NOT NULL,
                    virtual_sensor_id INT,
                    formula_id INT,
                    aqi_value FLOAT4 NOT NULL,
                    aqi_percent FLOAT4 NOT NULL,
                    aqi_zone VARCHAR(25) NOT NULL
                ) PARTITION by range (sampled_at);



    /* ------------ Dashboard / Main Entities ------------ */

    CREATE TABLE IF NOT EXISTS company (
        id SERIAL PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        address VARCHAR(100) NOT NULL,
        manager INT,
        alarm_email VARCHAR(500),
        alarm_phones VARCHAR(500),
        telephone VARCHAR(200),
        logo VARCHAR(200),
        created_at BIGINT NOT NULL,
        created_by INT NOT NULL,
        updated_at BIGINT,
        updated_by INT,
        deleted_at BIGINT,
        deleted_by INT
    );

    CREATE TABLE IF NOT EXISTS facility (
        id SERIAL PRIMARY KEY,
        company_id INT NOT NULL,
        name VARCHAR(100) NOT NULL,
        image_url VARCHAR(255),
        lat FLOAT8 NOT NULL,
        lon FLOAT8 NOT NULL,
        timezone varchar(50),
        city VARCHAR(50),
        region VARCHAR(50),
        country VARCHAR(100),
        created_at BIGINT NOT NULL,
        created_by INT NOT NULL,
        email VARCHAR(255),  -- email of created_by
        updated_at BIGINT,
        updated_by INT,
        deleted_at BIGINT,
        deleted_by INT,
        license_type VARCHAR(50),
        license_expiration_time BIGINT,
        preferred_weather_equipment_id INT
    );

    CREATE TABLE IF NOT EXISTS groups (
        id SERIAL PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        facility_id INT NOT NULL,
        company_id INT NOT NULL,
        created_at BIGINT NOT NULL,
        created_by INT NOT NULL,
        updated_at BIGINT,
        updated_by INT,
        deleted_at BIGINT,
        deleted_by INT,
        image_url VARCHAR(255),
        type VARCHAR(50),
        signal_factor_n REAL,
        signal_factor_p1 REAL,  -- for location calculation from signal strength, tentative
        lat FLOAT8,
        lon FLOAT8
    );

    CREATE TABLE IF NOT EXISTS subgroups (
        id SERIAL PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        type VARCHAR(50),
        company_id INT NOT NULL,
        facility_id INT NOT NULL,
        group_id INT NOT NULL,
        created_at BIGINT NOT NULL,
        created_by INT NOT NULL,
        updated_at BIGINT,
        updated_by INT,
        deleted_at BIGINT,
        deleted_by INT,
        image_url VARCHAR(255),
        color_id VARCHAR(50),
        lat FLOAT8,
        lon FLOAT8
    );
      CREATE TABLE IF NOT EXISTS hub (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            company_id INT NOT NULL,
            facility_id INT NOT NULL,
            group_id INT NOT NULL,
            created_at BIGINT NOT NULL,
            created_by INT NOT NULL,
            updated_at BIGINT,
            updated_by INT,
            deleted_at BIGINT,
            deleted_by INT,
            image_url VARCHAR(255),
            last_signaled_at BIGINT,
            signal_to_base INT  -- TODO: might not need
        );

    CREATE TABLE IF NOT EXISTS gateway_hub (
        id SERIAL PRIMARY KEY,
        name VARCHAR(100),
        mac_address VARCHAR(20) NOT NULL,  -- TODO: adjust length if needed, add serial_number if needed
        -- serial_number VARCHAR(20) NOT NULL,  -- TODO: uncomment if needed
        company_id INT NOT NULL,
        facility_id INT NOT NULL,
        group_id INT,  -- TODO: remove if not grouping by hub network as suggested in design doc
        lat FLOAT8,
        lon FLOAT8,
        alt FLOAT8,  -- remove if not necessary
        created_at BIGINT NOT NULL,
        created_by INT NOT NULL,
        updated_at BIGINT,
        updated_by INT,
        deleted_at BIGINT,
        deleted_by INT,
        image_url VARCHAR(255),
        -- last_signaled_at BIGINT,  -- TODO: uncomment if needed
        signal_to_base INT  -- TODO: in design doc, delete if not needed
    );

    -- sort of a many-many relationship table, 1 and only 1 entry for each <hub, equipment> pair
    CREATE TABLE IF NOT EXISTS hub_equipment (
        hub_id INT NOT NULL,
        equipment_id INT NOT NULL,
        signal_db INT,
        last_signaled_at BIGINT NOT NULL
    );
    CREATE INDEX hub_equipment_hub_id_equipment_id_idx ON hub_equipment (hub_id, equipment_id);


    CREATE TABLE IF NOT EXISTS equipment (
        -- Basic equipment info
        id SERIAL PRIMARY KEY,
        company_id INT NOT NULL,
        facility_id INT,  -- make facility and group nullable so that user could decide for new equipment
        group_id INT,
        subgroup_id INT,
        serial_number VARCHAR(255) NOT NULL,
        model VARCHAR(50) NOT NULL,
        status VARCHAR(50),
        created_by INT NOT NULL,
        created_at BIGINT NOT NULL,
        updated_at BIGINT,
        updated_by INT,
        deleted_at BIGINT,
        deleted_by INT,
        image_url VARCHAR(255),
        notification_count INT,
        name VARCHAR(100),
        secret VARCHAR(255),

        -- equipment location Info
        lat FLOAT, --nullable for iaq equipment
        lon FLOAT, --nullable for iaq equipment
        -- removed regional info, will only keep for facility

        -- equipment data and settings
        -- removed wind_speed and wind_direction
        last_sampled_at BIGINT,
        settings_updated_at BIGINT,
        settings JSONB,
        maintenance_interval INT,
        last_maintained_at BIGINT,
        last_calibrated_at BIGINT,
        license_type VARCHAR(50),
        license_expiration_time BIGINT,
        firmware VARCHAR(100)
    );
        CREATE TABLE IF NOT EXISTS hub (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            company_id INT NOT NULL,
            facility_id INT NOT NULL,
            group_id INT NOT NULL,
            created_at BIGINT NOT NULL,
            created_by INT NOT NULL,
            updated_at BIGINT,
            updated_by INT,
            deleted_at BIGINT,
            deleted_by INT,
            image_url VARCHAR(255),
            last_signaled_at BIGINT,
            signal_to_base INT  -- TODO: might not need
        );

    CREATE TABLE IF NOT EXISTS equipment_settings (
        id int PRIMARY KEY,
        settings_name VARCHAR(100)
    );

    CREATE TABLE IF NOT EXISTS equipment_alarms (
        id SERIAL PRIMARY KEY,
        code VARCHAR(10),
        message VARCHAR(500),
        category VARCHAR(100),
        type VARCHAR(50)
    );



    CREATE TABLE IF NOT EXISTS sensor (  -- reordered to match the cloud
        /* ------- General Fields  (Common fields for both) ------- */
        id SERIAL PRIMARY KEY,    -- 0
        type_id int NOT NULL,     -- 1
        equipment_id INT NOT NULL,  -- 2
        unit VARCHAR(50) NOT NULL,  -- 3
        port VARCHAR(50),           -- 4
        detection_range_max FLOAT8, -- 5
        detection_range_min FLOAT8, -- 6
        sensitivity FLOAT8,         -- 7
        zero_offset_voltage FLOAT8, -- 8
        calibration_factors VARCHAR(255),
        last_calibrated_at BIGINT,
        calibration_interval INT,
        last_sampled_value FLOAT4,
        aqi_limits_id INT,
        -- SIMS1 FIELDS WE HAVE JUST IN CASE
        associated_relay_number FLOAT8,
        relay_trigger_comparison FLOAT8,
        relay_trigger_limit FLOAT8,
        packet_id VARCHAR(50),  -- TODO: set to NOT NULL later
        last_updated_at BIGINT, -- field in sims2 for sensor setting
        last_sampled_at BIGINT,
        has_raw_data BOOL,  -- true if raw data is sent in and needs option for display
        created_at BIGINT,  -- TODO: setting to NOT NULL later
        deleted_at BIGINT,
        deleted_by INT,
        updated_by INT,
        created_by INT
    );

    CREATE INDEX sensor_equipment_id_packet_id_idx ON sensor (equipment_id, packet_id);

    CREATE TABLE IF NOT EXISTS sensor_type (
        id SERIAL PRIMARY KEY,
        full_name VARCHAR(50) NOT NULL,
        short_name VARCHAR(20) NOT NULL,
        category VARCHAR(20) NOT NULL,
        lowest_detection_limit VARCHAR(50),
        maximum_detection_limit VARCHAR(50),
        default_unit VARCHAR(10),
        molar_mass_gmol FLOAT8,
        unit_conversions VARCHAR(100),
        ppm_to_ug_m3_multiplier FLOAT8,
        ug_m3_to_ppm_multiplier FLOAT8,
        created_at BIGINT,
        created_by INT,
        updated_at BIGINT,
        updated_by INT,
        deleted_at BIGINT,
        deleted_by INT
    );

    CREATE TABLE IF NOT EXISTS aqi (
        id SERIAL PRIMARY KEY,
        company_id INT,
        facility_id INT,
        sensor_id INT,
        sensor_type_id INT,
        unit VARCHAR(50),
        odor_detection_threshold VARCHAR(50),  -- the threshold for detecting odour, or 'Odorless'
        averaging_period INT NOT NULL,
        good FLOAT8 NOT NULL,
        moderate FLOAT8 NOT NULL,
        sensitive FLOAT8 NOT NULL,
        unhealthy FLOAT8 NOT NULL,
        very_unhealthy FLOAT8 NOT NULL,
        hazardous FLOAT8 NOT NULL,
        created_at BIGINT,
        created_by INT,
        updated_at BIGINT,
        updated_by INT,
        deleted_at BIGINT,
        deleted_by INT
    );

    CREATE TABLE IF NOT EXISTS weather (
        id                  SERIAL,
        is_forecast         BOOL,                   -- true for forecast from /forecasts endpoint, false for current from /conditions,
        interval_type       SMALLINT,               -- 0 for hourly, 1 for daily
        fetched_at          BIGINT,                 -- timestamp when the weather record is fetched from AerisWeather
        city                VARCHAR(50),             -- place or nearest place to the weather record by AerisWeather
        is_day              BOOL,
        facility_id         INT           NOT NULL,
        facility_lat        FLOAT8        NOT NULL,  -- coordinates of facility
        facility_lon        FLOAT8        NOT NULL,
        time_epoch          BIGINT        NOT NULL,
        time_iso            VARCHAR(255)  NOT NULL,
        icon_name           VARCHAR(50)   NOT NULL,  -- image file name for the icon
        temperature_c       FLOAT4        NOT NULL,
        wind_direction      VARCHAR(10)   NOT NULL,  -- string version, e.g. "N", "NE"
        wind_direction_deg  INT           NOT NULL,  -- in degrees (0-359), 0 = North
        wind_speed_ms       FLOAT4        NOT NULL,
        cloud_cover         INT           NOT NULL,  -- percentage (0-100)
        solar_radiation_wm2 INT           NOT NULL,
        humidity            INT,                     -- percentage (0-100)
        pressure_hpa        INT,
        precipitation_mm    FLOAT8,
        probability_of_precipitation  INT,           -- percentage (0-100), probability of precipitation
        complaint_risk      VARCHAR(50)
    ) PARTITION BY RANGE (time_epoch);

    CREATE INDEX weather_facility_id_time_epoch_idx ON weather (facility_id, time_epoch);

    CREATE TABLE IF NOT EXISTS emission_source (
        id                SERIAL     PRIMARY KEY,
        company_id        INT,
        facility_id       INT           NOT NULL,
        preferred_weather_equipment_id  INT,  -- TODO: delete if not keeping any ES specificity
        created_by        INT           NOT NULL,
        created_at        BIGINT        NOT NULL,
        name              VARCHAR(50),
        source_shape       VARCHAR(50),  -- any of 'Point', 'Rectangular', 'Circular'
        monitor_option        VARCHAR(50),
        sensors           integer[],  -- list of sensor being monitored
        is_urban          BOOL          NOT NULL,  -- TRUE for urban, FALSE for rural
        center_lat               FLOAT8        NOT NULL,
        center_lon               FLOAT8        NOT NULL,
        latest_emission_rate_gs  FLOAT8        NOT NULL,
        last_updated_at   BIGINT,
        last_updated_by   INT,
        deleted_at BIGINT,
        deleted_by INT,
        image_url         VARCHAR(255),
        is_active         BOOL          NOT NULL,   -- true by default
        sensor_type       INT,
        emission_rate_unit  VARCHAR(50),
        source_configs    JSONB
        /* source_configs: store additional information specific for different source shape
        - Point: 'stack_height_m', 'stack_diameter_m', 'exit_temp_c', 'exit_velocity_ms'
        - Circular: 'center_xy', 'radius_m'
        - Rectangular: 'center_xy', 'height_m', 'width_m', 'theta_src', 'vertices_coordinates'
        */
    );

    CREATE TABLE IF NOT EXISTS plume (
        id                      SERIAL PRIMARY KEY,  -- might not be needed

        /* ------ from emission_source table ------ */
        facility_id             INT           NOT NULL,
        esource_id              INT           NOT NULL,  -- emission source id
        is_forecast             BOOL,                    -- True if it's based on forecast weather, False otherwise
        is_urban                BOOL          NOT NULL,  -- TRUE for urban, FALSE for rural
        plume_origin_lat               FLOAT8        NOT NULL,  -- stack lat and lon for point source; outermost strip's center for area source
        plume_origin_lon               FLOAT8        NOT NULL,  -- either way it'd be the dispersion's origin point for plotting the plume
        source_shape            VARCHAR(50)   NOT NULL,
        source_configs          JSONB,
        latest_emission_rate_gs        FLOAT8        NOT NULL,
        emission_rate_unit      VARCHAR(50)   NOT NULL,
        source_last_updated_at   BIGINT        NOT NULL,

        /* ------ possible preferred equipment -------- */
        preferred_weather_equipment_id  INT,
        equipment_last_sampled_at BIGINT,
        equipment_sample_id JSONB,

        /* ------ from weather table -------- */
        weather_id              INT       NOT NULL,
        wind_speed_ms           FLOAT4    NOT NULL,  -- in m/s
        wind_direction_deg      INT       NOT NULL,  -- degrees from North (0-360)
        cloud_cover             INT       NOT NULL,  -- percentage (0-100)
        solar_radiation_wm2     INT,                 -- in Watts/m^2
        ambient_temperature_c   FLOAT4    NOT NULL,  -- Celsius
        weather_time_epoch      BIGINT    NOT NULL,
        is_day                  BOOL      NOT NULL,

        concentration_max       FLOAT8    NOT NULL,  -- for reproducing grid from 8-bit normalized image
        image_name              TEXT      NOT NULL,  -- output plume image name/address
        produced_at             BIGINT    NOT NULL,
        produced_for            BIGINT               -- equals produced_at for current ones, and weather_time_epoch for forecast ones
    );

    CREATE TABLE IF NOT EXISTS triangulation_grid (
            id                  SERIAL PRIMARY KEY,  -- might not be needed

            /* ------ entity info for which the triangulation grid is calculated ------ */
            entity_type         VARCHAR(50)   NOT NULL,
            entity_id           INT           NOT NULL,
            entity_lat          FLOAT8        NOT NULL,  -- lat and lon of entity, which would also be the center of the triangulation grid
            entity_lon          FLOAT8        NOT NULL,
            facility_id         INT           NOT NULL,  -- facility id of the entity

            /* ------ inputs needed to compute the grid -------- */
            weather_entity_info   JSONB,                    -- contains record info for facility/equipment/weather table
            wind_direction_deg    INT           NOT NULL,   -- degrees from North (0-360)
            wind_speed_ms         FLOAT4        NOT NULL,   -- in m/s
            sigma_y               FLOAT8        NOT NULL,   -- calculated standard deviation
            -- downwind_x           INT,                      -- in meters, default 5000m
            psq                   VARCHAR(10),              -- calculated Pasquill stability class
            solar_radiation_wm2   INT,                      -- in Watts/m^2
            cloud_cover           INT,                      -- percentage (0-100)
            is_day                BOOL,                     -- True for during the day, False for night

            /* ------ Calculated grid -------- */
            grid_max              FLOAT8        NOT NULL,   -- for reproducing grid from 8-bit normalized image
            image_name            TEXT          NOT NULL,   -- grid image name (no path)
            produced_for          BIGINT        NOT NULL,   -- timestamp of the grid
            produced_at           BIGINT        NOT NULL    -- server time of execution
        );

    CREATE TABLE IF NOT EXISTS alarm_rule (
        id SERIAL PRIMARY KEY,
        sensor_id INT,
        type varchar(50),
        virtual_sensor_id INT,
        company_id INT NOT NULL,
        factor VARCHAR(100) NOT NULL,
        tag VARCHAR(100) NOT NULL,
        condition VARCHAR(100) NOT NULL,
        trigger VARCHAR(100) ,
        frequency INT ,
        created_by INT NOT NULL,
        user_list VARCHAR(500),
        phone_list VARCHAR(500)
    );

    CREATE TABLE IF NOT EXISTS event_queue (
        id VARCHAR(50),
        event_category VARCHAR(50) NOT NULL,
        company_id int NOT NULL,
        -- equipment_id int NOT NULL,
        equipment_id int,
        sensor_id int,
        event_id int NOT NULL,
        created_at BIGINT,
        updated_at BIGINT,
        triggered_at BIGINT,
        day date NOT NULL DEFAULT CURRENT_DATE,
        alarm_ext_notification_sent_at BIGINT,
        PRIMARY KEY(id, day)
    ) PARTITION by range (day);

    CREATE TABLE IF NOT EXISTS partitions.event_queue_20211101 partition of event_queue for values from ('2021-11-01') to ('2021-12-01');
    CREATE TABLE IF NOT EXISTS partitions.event_queue_20211201 partition of event_queue for values from ('2021-12-01') to ('2022-01-01');
    CREATE TABLE IF NOT EXISTS partitions.event_queue_20220101 partition of event_queue for values from ('2022-01-01') to ('2022-02-01');
    CREATE TABLE IF NOT EXISTS partitions.event_queue_20220201 partition of event_queue for values from ('2022-02-01') to ('2022-03-01');
    CREATE TABLE IF NOT EXISTS partitions.event_queue_20220301 partition of event_queue for values from ('2022-03-01') to ('2022-04-01');
    CREATE TABLE IF NOT EXISTS partitions.event_queue_20220401 partition of event_queue for values from ('2022-04-01') to ('2022-05-01');
    CREATE TABLE IF NOT EXISTS partitions.event_queue_20220501 partition of event_queue for values from ('2022-05-01') to ('2022-06-01');

    CREATE TABLE IF NOT EXISTS event (
        id SERIAL ,
        category VARCHAR(100) NOT NULL,
        company_id INT NOT NULL,
        equipment_id INT,
        sensor_id INT,
        sensitive_receptor_id INT,
        virtual_equipment_id INT,
        virtual_sensor_id INT,
        emission_source_id INT,
        description VARCHAR(250) NOT NULL,
        admin_note VARCHAR(250),
        occurrences INT NOT NULL,
        created_by INT NOT NULL,
        created_at BIGINT NOT NULL,
        updated_at BIGINT NOT NULL,
        justification VARCHAR(250) NOT NULL,
        justification_time BIGINT,
        justification_type VARCHAR(250),
        justification_concentration JSONB,
        user_type VARCHAR(250),
        username VARCHAR(250),
        sources integer[],
        is_hidden BOOL NOT NULL,
        intensity INT,
        guest_id VARCHAR(25), --using for complaint app document storage
        lat FLOAT8,
        lon FLOAT8,
        duration VARCHAR(255),
        started_at BIGINT,
        finished_at BIGINT,
        facility_id INT,
        group_id INT,
        subgroup_id INT,
        alarm_id INT,
        email_list VARCHAR(250),
        message VARCHAR(500),
        day date NOT NULL DEFAULT CURRENT_DATE,
        PRIMARY KEY(id , day),
        category_id INT
    ) PARTITION BY range  (day);

    CREATE TABLE IF NOT EXISTS partitions.event_20211101 partition of event for values from ('2021-11-01') to ('2021-12-01');
    CREATE TABLE IF NOT EXISTS partitions.event_20211201 partition of event for values from ('2021-12-01') to ('2022-01-01');
    CREATE TABLE IF NOT EXISTS partitions.event_20220101 partition of event for values from ('2022-01-01') to ('2022-02-01');
    CREATE TABLE IF NOT EXISTS partitions.event_20220201 partition of event for values from ('2022-02-01') to ('2022-03-01');
    CREATE TABLE IF NOT EXISTS partitions.event_20220301 partition of event for values from ('2022-03-01') to ('2022-04-01');
    CREATE TABLE IF NOT EXISTS partitions.event_20220401 partition of event for values from ('2022-04-01') to ('2022-05-01');
    CREATE TABLE IF NOT EXISTS partitions.event_20220501 partition of event for values from ('2022-05-01') to ('2022-06-01');

    CREATE TABLE IF NOT EXISTS notification (
        event_id INT NOT NULL,
        company_id INT NOT NULL,
        viewed_at BIGINT,
        viewed_by INT,
        /* - week is used to partition table by week - */
        day date NOT NULL DEFAULT CURRENT_DATE,
        PRIMARY KEY (event_id, company_id, day)
    ) PARTITION BY range  (day);
    CREATE TABLE IF NOT EXISTS partitions.notification_20211101 partition of notification for values from ('2021-11-01') to ('2021-12-01');
    CREATE TABLE IF NOT EXISTS partitions.notification_20211201 partition of notification for values from ('2021-12-01') to ('2022-01-01');
    CREATE TABLE IF NOT EXISTS partitions.notification_20220101 partition of notification for values from ('2022-01-01') to ('2022-02-01');
    CREATE TABLE IF NOT EXISTS partitions.notification_20220201 partition of notification for values from ('2022-02-01') to ('2022-03-01');
    CREATE TABLE IF NOT EXISTS partitions.notification_20220301 partition of notification for values from ('2022-03-01') to ('2022-04-01');
    CREATE TABLE IF NOT EXISTS partitions.notification_20220401 partition of notification for values from ('2022-04-01') to ('2022-05-01');
    CREATE TABLE IF NOT EXISTS partitions.notification_20220501 partition of notification for values from ('2022-05-01') to ('2022-06-01');


    CREATE TABLE IF NOT EXISTS event_category (
        category VARCHAR(100) NOT NULL,
        company_id INT NOT NULL,
        event_defined_by VARCHAR(100) NOT NULL,
        affiliated_entity VARCHAR(255) NOT NULL,
        created_by INT NOT NULL,
        created_at BIGINT NOT NULL,
        description VARCHAR(255),
        threshold_justification_value FLOAT8,
        PRIMARY KEY(category, company_id)
    );

    CREATE TABLE IF NOT EXISTS extraction_queue_bulk (
        extraction_id SERIAL,
        start_at BIGINT NOT NULL,
        end_at BIGINT NOT NULL,
        equipment_list integer[] NOT NULL,
        equipment_sensors jsonb NOT NULL,
        created_at BIGINT NOT NULL,
        created_by INT NOT NULL,
        company_id INT,
        file_type VARCHAR(50),
        timezone varchar(25),
        extracted_at BIGINT,
        day date NOT NULL DEFAULT CURRENT_DATE,
        status INT,
        data_url VARCHAR(255),
        PRIMARY KEY(extraction_id , day)
    )  PARTITION BY range (day);

    CREATE TABLE IF NOT EXISTS partitions.extraction_queue_bulk_20231001 partition of extraction_queue_bulk for values from ('2023-10-01') to ('2023-11-01');

    CREATE TABLE IF NOT EXISTS extraction_queue (
        extraction_id SERIAL,
        start_at BIGINT NOT NULL,
        end_at BIGINT NOT NULL,
        sensor_list integer[],
        virtual_sensor_list integer[],
        event_id int,
        sensor_unit_list JSONB, --Eventually will use this instead of sensor list
        extracted_at BIGINT,
        day date NOT NULL DEFAULT CURRENT_DATE,
        status INT,
        data_url VARCHAR(255),
        file_type VARCHAR(50),
        created_at BIGINT NOT NULL,
        created_by INT NOT NULL,
        company_id INT,
        timezone varchar(25),
        report_id INT,
        email_list varchar(500),
        new_report_id varchar(50),
        PRIMARY KEY(extraction_id , day),
        description VARCHAR(255)
    )  PARTITION BY range (day);

    CREATE TABLE IF NOT EXISTS partitions.extraction_queue_20220201 partition of extraction_queue for values from ('2022-02-01') to ('2022-03-01');
    CREATE TABLE IF NOT EXISTS partitions.extraction_queue_20220301 partition of extraction_queue for values from ('2022-03-01') to ('2022-04-01');
    CREATE TABLE IF NOT EXISTS partitions.extraction_queue_20220401 partition of extraction_queue for values from ('2022-04-01') to ('2022-05-01');
    CREATE TABLE IF NOT EXISTS partitions.extraction_queue_20220501 partition of extraction_queue for values from ('2022-05-01') to ('2022-06-01');

    CREATE TABLE IF NOT EXISTS migration_id_lookup (
        old_id INT NOT NULL,
        sims3_id INT NOT NULL,
        table_name VARCHAR(100) NOT NULL,
        database VARCHAR(100) NOT NULL
    );

    CREATE TABLE IF NOT EXISTS source_estimation (
       id serial,
       emission_source_id INT,
       sensor_and_values JSONB,
       emission_source_info JSONB,
       source_type VARCHAR(50),
       monitor_option VARCHAR(50),
       previous_emission_rate FLOAT8,
       calculated_emission_rate FLOAT8,
       discard_reason TEXT, -- null if calculated, if not null, the calculation was skipped, and entry was directly copied from previous record
       produced_at BIGINT,  -- actual execution time
       calculated_at BIGINT  -- produced_for, every 10 min
    ) PARTITION BY RANGE (calculated_at);

    CREATE TABLE IF NOT EXISTS calibration_records (
        id SERIAL PRIMARY KEY,
        calibrated_on BIGINT NOT NULL,
        calibration_type VARCHAR(50),
        sensitivities float4,
        zero_offset float4,
        sensor_id INT NOT NULL,
        equipment_id INT NOT NULL, -- not necessary but makes querying a bit easier
        created_by INT NOT NULL,  -- same person who calibrated it
        created_at BIGINT NOT NULL,
        deleted_at BIGINT,  -- Because these are important records we will not delete forever
        deleted_by INT
    );

    -- sensitive_receptor is included in table groups with a '"sR_group"' type, thus removing the table

    CREATE TABLE IF NOT EXISTS virtual_equipment (
      id SERIAL PRIMARY KEY,
      name VARCHAR(50) NOT NULL,
      lat FLOAT8 NOT NULL,
      lon FLOAT8 NOT NULL,
      company_id INT NOT NULL,
      facility_id INT NOT NULL,
      group_id INT NOT NULL,
      model VARCHAR(50) NOT NULL,
      image_url VARCHAR(255),
      created_at BIGINT NOT NULL,
      created_by INT NOT NULL,
      deleted_at BIGINT,
      deleted_by INT,
      sources integer[],
      last_sampled_at BIGINT,
      updated_at BIGINT,
      included Boolean NOT NULL
    );

    CREATE TABLE IF NOT EXISTS virtual_sensor (
          id SERIAL PRIMARY KEY,
          name VARCHAR(100) NOT NULL,
          equipment_id INT,
          real_equipment_id INT,
          unit VARCHAR(50),
          last_sampled_value FLOAT8,
          last_sampled_at BIGINT,
          aqi_limits_id INT,
          formula_id INT,
          short_name VARCHAR(50) NOT NULL,
          sensor_type_id INT,
          emission_source INT,
          emission_type VARCHAR(50),
          is_active BOOL,
          created_at BIGINT NOT NULL,
          created_by INT NOT NULL,
          deleted_at BIGINT,
          deleted_by INT,
          updated_at BIGINT
        );

    CREATE TABLE IF NOT EXISTS virtual_samples (
      id serial,
      sampled_at BIGINT NOT NULL,
      virtual_equipment_id INT,
      lat FLOAT8,
      lon FLOAT8,
      sensor_value FLOAT4,
      virtual_sensor_id INT,
      is_predicted BOOL,
      formula_id int,
      real_equipment_id int
    ) PARTITION by range (sampled_at);

    CREATE TABLE IF NOT EXISTS travel_session (
        id SERIAL PRIMARY KEY,
        equipment_id INT NOT NULL,
        start_timestamp BIGINT NOT NULL,
        end_timestamp BIGINT NOT NULL,
        distance FLOAT8,
        timezone VARCHAR(20) NOT NULL,
        created_at BIGINT,
        sample_file VARCHAR(100),
        sample_file_processed_at BIGINT
    );

    CREATE TABLE IF NOT EXISTS report(
        id SERIAL PRIMARY KEY,
        type VARCHAR(50) NOT NULL,
        frequency VARCHAR(20) NOT NULL, --how often report is run
        sensor_unit_list JSONB,
        last_report BIGINT,
        next_report BIGINT,
        company_id int NOT NULL,
        start_time BIGINT, --for one time reports
        end_time BIGINT, --for one time reports,
        timezone varchar(25) NOT NULL,
        created_at BIGINT,
        created_by INT
    );

    CREATE TABLE IF NOT EXISTS report_new(
            id VARCHAR(50) PRIMARY KEY,
            type VARCHAR(50) NOT NULL,
            frequency VARCHAR(20),
            focus VARCHAR(50),
            sensor_unit_list JSONB,
            equipment_list integer[],
            last_report BIGINT,
            next_report BIGINT,
            company_id int NOT NULL,
            event_id INT,
            facility_id int,
            start_time BIGINT, --for one time reports
            end_time BIGINT, --for one time reports,
            timezone varchar(25) NOT NULL,
            created_at BIGINT,
            created_by INT,
            email_list VARCHAR(500),
            next_processed_time BIGINT,
            last_processed_time BIGINT,
            deleted_at BIGINT,
            deleted_by INT,
            description VARCHAR(500)
        );


    CREATE TABLE IF NOT EXISTS complaint_risk(
       id SERIAL,
       calculated_at BIGINT,
       calculated_for BIGINT,
       virtual_sensor_id int,
       max_concentration FLOAT8,
       facility_id int,
       complaint_risk_percentage FLOAT8
    ) PARTITION by range (calculated_for);

    Create TABLE IF NOT EXISTS formula (
      id SERIAL PRIMARY KEY,
      full_name VARCHAR(100) NOT NULL,
      short_name VARCHAR(20) NOT NULL,
      string_formula VARCHAR(250) NOT NULL,
      sensor_id_list  integer[] NOT NULL,
      equipment_id int NOT NULL,
      facility_id int  NOT NULL,
      company_id int  NOT NULL,
      is_active bool,
      last_calculated_at BIGINT,
      last_calculated_value float,
      aqi_limits_id INT,
      created_at BIGINT NOT NULL,
      created_by INT NOT NULL,
      deleted_at BIGINT,
      deleted_by INT
    );

    Create TABLE IF NOT EXISTS unit_sig_figs (
          id SERIAL PRIMARY KEY,
          unit VARCHAR(100) NOT NULL,
          sig_fig int NOT NULL
    );

    Create TABLE IF NOT EXISTS maintenance_types (
      id SERIAL PRIMARY KEY,
      model VARCHAR(50) NOT NULL,
      description VARCHAR(255) NOT NULL,
      frequency BIGINT NOT NULL,
      threshold BIGINT NOT NULL,
      created_by INT,
      created_at BIGINT
    );

    Create TABLE IF NOT EXISTS maintenance_records (
      id SERIAL PRIMARY KEY,
      equipment_id INT NOT NULL,
      facility_id INT NOT NULL,
      group_id INT NOT NULL,
      company_id INT NOT NULL,
      completed_at BIGINT,
      is_completed BOOL,
      created_by INT,
      created_at BIGINT,
      maintenance_type_id INT NOT NULL
    );

    Create TABLE IF NOT EXISTS auto_zero_sensors (
      id SERIAL PRIMARY KEY,
      sensor_id INT NOT NULL,
      equipment_id INT NOT NULL,
      frequency varchar(10) NOT NULL,
      target_min float NOT NULL,
      comparator varchar(10) NOT NULL default 'min',
      rate_of_change_limit float NOT NULL,
      absolute_change_limit float NOT NULL,
      validity_criteria float NOT NULL default 90,
      created_at bigint NOT NULL,
      created_by int not NULL,
      updated_at bigint,
      updated_by int,
      last_calculated_at bigint
    );



  COMMIT;
EOSQL
